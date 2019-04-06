import torch
import time
import datasets
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import sklearn
from sklearn.cluster import KMeans

def kNN(epoch, net, trainloader, testloader, K, sigma, ndata, low_dim = 128):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    correct_t = 0
    testsize = testloader.dataset.__len__()

   
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        try:
            trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()
        except:
            trainLabels = torch.LongTensor(trainloader.dataset.labels).cuda()
    trainFeatures = np.zeros((low_dim, ndata))    
    C = trainLabels.max() + 1
    C = np.int(C)
    with torch.no_grad():
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=4)
        for batch_idx, (inputs, _, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            # 
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
            
    trainloader.dataset.transform = transform_bak
    # 
    
    trainFeatures = torch.Tensor(trainFeatures).cuda()    
    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)  
            features = net(inputs)
            total += targets.size(0)

            
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()


            print('Test [{}/{}]\t'
              'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
              'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
              'Top1: {:.2f}  Top5: {:.2f}'.format(
              total, testsize, top1*100./total, top5*100./total, net_time=net_time, cls_time=cls_time))

    print(top1*100./total)

    return top1*100./total 
    
def eval_nmi_recall(epoch, net, lemniscate, testloader, feature_dim = 128):    
    net.eval()
    net_time = AverageMeter()
    val_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()
    
    ptr =0
    nmi = 0.
    recal = 0.
    end = time.time()
    test_features = np.zeros((testsize,feature_dim))
    test_labels   = np.zeros(testsize)
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            # end = time.time()
            batchSize = inputs.size(0)
            real_size = min(batchSize, testsize-ptr)
            targets = np.asarray(targets)
            batch_features = net(inputs)
            test_features[ptr:ptr+real_size,:] = batch_features
            test_labels[ptr:ptr+real_size]  = targets
            ptr += real_size
    net_time.update(time.time() - end)
    print('Extracting Time:\t'
                  'Net Time {net_time.val:.3f}s \t'
                  .format(net_time=net_time))
    # 
    # print('Evaluating.....')
    end = time.time()
    recal = eval_recall(test_features,test_labels)
    nmi = eval_nmi(test_features, test_labels)
    val_time.update(time.time() - end)
    print('Evaluating Time:\t'
                  'Eval Time {val_time.val:.3f}s \t'
                  .format(val_time=val_time))
    return recal, nmi 
    
def eval_recall(embedding, label):
    norm = np.sum(embedding*embedding,axis = 1)
    right_num = 0
    for i in range(embedding.shape[0]):
        dis = norm[i] + norm - 2*np.squeeze(np.matmul(embedding[i],embedding.T))
        dis[i] = 1e10
        pred = np.argmin(dis)
        if label[i]==label[pred]:
            right_num = right_num+1
    recall = float(right_num)/float(embedding.shape[0])
    return recall

def eval_nmi(embedding, label,  normed_flag = False, fast_kmeans = False):
    unique_id = np.unique(label)
    num_category = len(unique_id)
    if normed_flag:
        for i in range(embedding.shape[0]):
            embedding[i,:] = embedding[i,:]/np.sqrt(np.sum(embedding[i,:] ** 2)+1e-4)
    if fast_kmeans:
        kmeans = KMeans(n_clusters=num_category, n_init = 1, n_jobs=8)
    else:
        kmeans = KMeans(n_clusters=num_category,n_jobs=8)
    kmeans.fit(embedding)
    y_kmeans_pred = kmeans.predict(embedding)
    nmi = normalized_mutual_info_score(label, y_kmeans_pred)
    return nmi

def eval_recall_K(embedding, label, K_list =None):
    if K_list is None:
        K_list = [1, 2, 4, 8]
    norm = np.sum(embedding*embedding,axis = 1)
    right_num = 0

    recall_list = np.zeros(len(K_list))

    for i in range(embedding.shape[0]):
        dis = norm[i] + norm - 2*np.squeeze(np.matmul(embedding[i],embedding.T))
        dis[i] = 1e10
        index = np.argsort(dis)
        list_index = 0
        for k in range(np.max(K_list)):
            if label[i]==label[index[k]]:
                recall_list[list_index] = recall_list[list_index]+1
                break
            if k>=K_list[list_index]-1:
                list_index = list_index + 1
    recall_list = recall_list/float(embedding.shape[0])
    for i in range(recall_list.shape[0]):
        if i == 0:
            continue
        recall_list[i] = recall_list[i]+recall_list[i-1]
    return recall_list
    
class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()      
