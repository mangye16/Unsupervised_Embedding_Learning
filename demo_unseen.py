'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time

import models
import datasets
import math
import numpy as np

from BatchAverage import BatchAverage, BatchCriterion
from utils import *

from tensorboardX import SummaryWriter


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch CUB200 Training')
parser.add_argument('--dataset', default='cub200',  help='dataset name')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--log_dir', default='log/', type=str, 
                    help='log save path')
parser.add_argument('--model_dir', default='checkpoint/', type=str, 
                    help='model save path')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--arch', '-a', metavar='ARCH', default='inception_v1_ml',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: inception_v1_ml)')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--batch-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')               
parser.add_argument('--scratch', dest='scratch', action='store_true',
                    help='training from the sratch')
parser.add_argument('--batch-size', default=64, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--batch-m', default=1, type=int,
                    metavar='N', help='m for negative sum') 
parser.add_argument('--ptr', default=0, type=int,
                    metavar='P', help='instance or class knn')    
parser.add_argument('--test-batch', default=100, type=int,
                    help='training batch size')                     
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')

src_dir = '/home/datasets/prml/computervision/classification/'                     

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

freeze_bn = 0

if args.dataset =='cub200':
    dataset = 'cub'
    K_list = [1,2,4,8]
elif args.dataset =='car196':
    dataset = 'car'
    K_list = [1,2,4,8]
elif args.dataset =='ebay':
    dataset = 'ebay'
    K_list = [1,10,100]

if not args.scratch:
    suffix = dataset + '_batch_{}_pret'.format(args.batch_size)
else:  
    suffix = dataset + '_batch_{}_scra'.format(args.batch_size)
    
suffix = suffix + '_temp_{}_km_{}_{}_alr'.format(args.batch_t, args.batch_m, args.arch)


log_dir = args.log_dir + dataset + '_log/'
checkpoint_path = args.model_dir
vis_log_dir = log_dir + suffix + '/'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)  
test_log_file = open(log_dir + suffix + '.txt', "w")       

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')

# Data loading code
if args.arch =='inception_v1_ml':
    normalize = transforms.Compose([
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x * 255.0),
	transforms.Normalize(mean = [122.7717, 115.9465, 102.9801], std = [1, 1, 1]),
	transforms.Lambda(lambda x: x[[2, 1, 0], ...])
    ])

    transform_train= transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(size=227),
        transforms.RandomHorizontalFlip(),
        normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(227),
        normalize,
    ])
else:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


trainset = datasets.MLDataInstance(src_dir = src_dir, dataset_name = args.dataset, train=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last =True)

testset = datasets.MLDataInstance(src_dir = src_dir, dataset_name = args.dataset, train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=4)

ndata = trainset.__len__()

# define model
print('==> Building model..')
if not args.scratch:
    print("=> using pre-trained model '{}'".format(args.arch))
    net = models.__dict__[args.arch](pretrained=True,low_dim=args.low_dim)
else:
    print("=> creating model '{}'".format(args.arch))
    net = models.__dict__[args.arch](low_dim=args.low_dim)

if args.arch =='resnet18':
    pool_dim = 512
elif args.arch=='inception_v1_ml':
    pool_dim = 1024
elif args.arch=='resnet50':
    pool_dim = 2048

# define leminiscate: inner product within each mini-batch (Ours)
lemniscate = BatchAverage(args.low_dim, args.batch_t, args.batch_size)

if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True

# define loss function: inner product loss within each mini-batch
criterion = BatchCriterion(args.batch_m, args.batch_size)

net.to(device)
lemniscate.to(device)
criterion.to(device)

if args.test_only or len(args.resume)>0:
    # Load checkpoint.
    model_path = args.model_dir + args.resume
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.model_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    lemniscate = checkpoint['lemniscate']
    best_acc = checkpoint['recall']
    start_epoch = checkpoint['epoch']
    
# define optimizer
optimizer = optim.SGD( net.parameters() , lr=args.lr, momentum=0.9, weight_decay=5e-4)



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 40:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
    writer.add_scalar('lr',  lr, epoch)
    

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    
    if freeze_bn:
        net.apply(set_bn_to_eval)
    
    end = time.time()
    for batch_idx, (inputs1, inputs2, targets, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs1, inputs2, targets, indexes = inputs1.to(device), inputs2.to(device), targets.to(device), indexes.to(device)
        
        inputs = torch.cat((inputs1,inputs2), 0)  
        optimizer.zero_grad()

        features = net(inputs)
        
        outputs = lemniscate(features, indexes)

        loss = criterion(outputs, indexes)
        
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

        # if batch_idx%20 ==0:
        print('Epoch: [{}][{}/{}]'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '.format(
              epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
    writer.add_scalar('loss',  train_loss.avg, epoch)
    
def compute_knn(dist_feat, trainLabels, knn=5,  epoch = 8):
    ''' compute the knn according to instance id/ class id
    '''
    ndata = len(trainLabels)   
    nnIndex = np.arange(ndata)
    
    top_acc = 0.
    # compute the instance knn
    for i in xrange(ndata):
        dist_feat[i,i] = -1000
        dist_tmp = dist_feat[i,:]
        ind = np.argpartition(dist_tmp, -knn)[-knn:]
        # random 1nn and augmented sample for positive 
        nnIndex[i] = np.random.choice([ind[0],i])
    return nnIndex.astype(np.int32)
   
for epoch in range(start_epoch, start_epoch+60):
    
    # generate positive index
    if args.dataset =='cub200':
        print('Extracting Train Features.....')
        net.eval()
        net_time = AverageMeter()
        val_time = AverageMeter()

        ptr =0
        end = time.time()
        transform_bak = trainset.transform
        trainset.transform = testloader.dataset.transform
        trainset.nnIndex = None 
        temploader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        train_features = np.zeros((ndata,pool_dim))
        trainLabels   = np.zeros(ndata)
        with torch.no_grad():
            for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
                batchSize = inputs.size(0)
                real_size = min(batchSize, args.test_batch)
                targets   = np.asarray(targets)
                _, batch_feat = net(inputs)
                train_features[ptr:ptr+real_size,:]  = np.asarray(batch_feat)
                ptr += args.test_batch
        net_time.update(time.time() - end)
        trainLabels = np.asarray(temploader.dataset.img_label)
        trainset.transform = transform_bak

        print('Extracting Time:\t'
                      'Net Time {net_time.val:.3f}s \t'
                      .format(net_time=net_time))
        # 
        print('Evaluating.....')
        end = time.time()
        
        # select nn Index
        dist_feat  = np.matmul(train_features, train_features.T)
        nn_index = compute_knn(dist_feat, trainLabels, knn=1, epoch = epoch)
    else:
        nn_index = np.arange(ndata)
        
    trainloader.dataset.nnIndex = nn_index
    
    # training
    train(epoch)
    
    # testing performance
    print('Extracting Test Features.....') 
    net.eval()    
    ptr =0
    end = time.time()
    test_size = testloader.dataset.__len__()
    test_features = np.zeros((test_size,args.low_dim))
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            batchSize = inputs.size(0)
            real_size = min(batchSize, args.test_batch)
            targets   = np.asarray(targets)

            batch_feat, _ = net(inputs)
            test_features[ptr:ptr+real_size,:] = np.asarray(batch_feat)
            ptr += real_size
    testLabels = np.asarray(testloader.dataset.img_label)
    print("Extracting Time: '{}'s".format(time.time()-end))
                  
    print('Evaluating.....')
    end = time.time()
    
    # compute recall at 1
    recal = eval_recall(test_features,testLabels)
    if args.dataset =='ebay' and epoch%10==0:
        nmi = eval_nmi(test_features, testLabels, fast_kmeans =True)
    else:
        nmi = eval_nmi(test_features, testLabels)

    print("Extracting Time: '{}'s".format(time.time()-end))

    writer.add_scalar('recall@1', recal, epoch)
    writer.add_scalar('nmi', nmi, epoch)
    print('[Epoch]: {}'.format(epoch))
    print('recall: {:.2%}'.format(recal))
    print('NMI: {:.2%}'.format(nmi))  

    if recal > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'lemniscate': lemniscate,
            'recall': recal,
            'nmi': nmi,
            'epoch': epoch,
        }
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        torch.save(state, checkpoint_path + suffix + '_best.t')
        if args.dataset =='cub200' or args.dataset =='car196' :
            recall_k = eval_recall_K(test_features,testLabels,K_list)
        best_acc = recal
        best_nmi = nmi
    print('best recall: {:.2f},   nmi: {:.2f}'.format(best_acc*100, best_nmi*100))

    
    print('[Epoch]: {}'.format(epoch), file = test_log_file)
    print('recall1: {:.2%} \t NMI: {:.2%}'.format(recal, nmi), file = test_log_file)
    if args.dataset =='ebay':
        print('(Best Recall @1  NMI:{})'.format(
            best_acc,  best_nmi), file = test_log_file)
    else:
        print('(Best Recall @1:{:.2%} \t @2:{:.2%} \t  @4:{:.2%} \t  @8:{:.2%} \t NMI:{})'.format(
            recall_k[0], recall_k[1],recall_k[2], recall_k[3], nmi), file = test_log_file)
    test_log_file.flush()

