'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import numpy as np
import models
import datasets
import math

from BatchAverage import BatchCriterion
from utils import *
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--dataset', default='cifar',  
                    help='dataset name: "cifar": cifar-10 datasetor "stl": stl-10 dataset]')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--log_dir', default='log/', type=str, 
                    help='log save path')
parser.add_argument('--model_dir', default='checkpoint/', type=str, 
                    help='model save path')
parser.add_argument('--test_epoch', default=1, type=int,
                    metavar='E', help='test every N epochs')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--batch-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--batch-m', default=1, type=float,
                    metavar='N', help='m for negative sum')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--gpu', default='0,1,2,3', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
args = parser.parse_args() 

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
 
dataset = args.dataset
if dataset =='cifar':
    img_size = 32
    pool_len = 4
elif dataset == 'stl':
    img_size = 96
    pool_len = 7
    
    
log_dir = args.log_dir + dataset + '_log/'
test_epoch = args.test_epoch
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
    
suffix = dataset + '_batch_0nn_{}'.format(args.batch_size)
suffix = suffix + '_temp_{}_km_{}_alr'.format(args.batch_t, args.batch_m)
    
if len(args.resume)>0:
    suffix = suffix + '_r'

# log the output
test_log_file = open(log_dir + suffix + '.txt', "w")                
vis_log_dir = log_dir + suffix + '/'
if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)  

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data Preparation
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=img_size, scale=(0.2,1.)),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if dataset =='cifar':
    # cifar-10 dataset 
    trainset = datasets.CIFAR10Instance(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last =True)

    testset = datasets.CIFAR10Instance(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, 
        batch_size=100, shuffle=False, num_workers=4)
elif dataset == 'stl':
    # stl-10 dataset 
    trainset = datasets.STL10Instance(root='./data', split='train+unlabeled', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last =True)

    valset = datasets.STL10Instance(root='./data', split='train', download=True, transform=transform_test)
    valloader = torch.utils.data.DataLoader(valset, 
        batch_size=100, shuffle=False, num_workers=4,drop_last =True)
    
    nvdata = valset.__len__()
    testset = datasets.STL10Instance(root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, 
        batch_size=100, shuffle=False, num_workers=4)

ndata = trainset.__len__()

print('==> Building model..')
net = models.__dict__['ResNet18'](pool_len = pool_len, low_dim=args.low_dim)

# define leminiscate: inner product within each mini-batch (Ours)

if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

# define loss function: inner product loss within each mini-batch
criterion = BatchCriterion(args.batch_m, args.batch_t, args.batch_size)

net.to(device)
criterion.to(device)

if args.test_only or len(args.resume)>0:
    # Load checkpoint.
    model_path = args.model_dir + args.resume
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.model_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
if args.test_only:
    if dataset == 'cifar':
        acc = kNN(epoch, net, trainloader, testloader, 200, args.batch_t, ndata, low_dim = args.low_dim)
    elif dataset == 'stl':
        acc = kNN(epoch, net, valloader, testloader, 200, args.batch_t, nvdata, low_dim = args.low_dim)
    sys.exit(0)
    
# define optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed at 120, 160 and 200"""
    lr = args.lr
    if epoch >= 120 and epoch < 160:
        lr = args.lr * 0.1
    elif epoch >= 160 and epoch <200:
        lr = args.lr * 0.05
    elif epoch >= 200:
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

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs1, inputs2, _, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)

        inputs1, inputs2, indexes = inputs1.to(device), inputs2.to(device), indexes.to(device)
        
        inputs = torch.cat((inputs1,inputs2), 0)
        optimizer.zero_grad()

        features = net(inputs)
        loss = criterion(features, indexes)

        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), inputs.size(0))         
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx%10 ==0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                  epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
    # add log
    writer.add_scalar('loss',  train_loss.avg, epoch)
    
for epoch in range(start_epoch, start_epoch+301):
    
    # training 
    train(epoch)
    
    # testing every test_epoch
    if epoch%test_epoch ==0:
        net.eval()
        print('----------Evaluation---------')
        start = time.time()
        
        if dataset == 'cifar':
            acc = kNN(epoch, net, trainloader, testloader, 200, args.batch_t, ndata, low_dim = args.low_dim)
        elif dataset == 'stl':
            acc = kNN(epoch, net, valloader, testloader, 200, args.batch_t, nvdata, low_dim = args.low_dim)
        
        print("Evaluation Time: '{}'s".format(time.time()-start))
        writer.add_scalar('nn_acc', acc, epoch)

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(args.model_dir):
                os.mkdir(args.model_dir)
            torch.save(state, args.model_dir + suffix + '_best.t')
            best_acc = acc
            
        print('accuracy: {}% \t (best acc: {}%)'.format(acc,best_acc))
        print('[Epoch]: {}'.format(epoch), file = test_log_file)
        print('accuracy: {}% \t (best acc: {}%)'.format(acc,best_acc), file = test_log_file)
        test_log_file.flush()
