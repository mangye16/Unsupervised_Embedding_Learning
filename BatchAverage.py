import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np
eps = 1e-7

class BatchAverage(nn.Module):
    ''' Compute the inner product within each batch  
    '''
    def __init__(self, FeatureDim, T=1, batchSize=128):       
        super(BatchAverage, self).__init__()
        self.T = T
        # initialize the memory (no use for our method)
        stdv = 1 / math.sqrt(FeatureDim)
        
        idx = GenerateIdx(batchSize*2)
        self.idx = torch.LongTensor(idx).cuda()

    def forward(self, feat,y):
        batchSize  = feat.size(0)
        featureDim = feat.size(1)
        
        # take out the batch feat according to the index
        weight = torch.index_select(feat.data, 0, self.idx.view(-1)) 
        
        weight = torch.reshape(weight, (batchSize, batchSize-1, featureDim))
        feat   = torch.reshape(feat, (batchSize, featureDim, 1))
        
        #inner product
        out = torch.bmm(weight, feat) # The first column is positive
        
        
        out.div_(self.T).exp_() # batchSize * batchSize

        feat = torch.reshape(feat, (batchSize, featureDim))
        out  = torch.reshape(out, (batchSize, batchSize-1))

        return out


class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch  
    '''
    def __init__(self, negM, batchSize =128):
        super(BatchCriterion, self).__init__()
        self.negM  = negM
        self.negNum = int(2* batchSize*self.negM)

    def forward(self, x, targets):
        batchSize = x.size(0)

        if not self.negM<1:
            # denominator: sum of the exp weights
 
            # positive probability
            Pmt = x.select(1,0)
            Pmt_div = x.narrow(1,1,batchSize-2).sum(1) * self.negM + Pmt
            lnPmt = torch.div(Pmt, Pmt_div)
            
            # negative probability
            Pon_div = Pmt_div.repeat(batchSize-2,1)
            Pon = x.narrow(1,1,batchSize-2)
            lnPon = torch.div(Pon, Pon_div.t())
            lnPon = -lnPon.add(-1)
            
            lnPmt.log_()
            lnPon.log_()
            
            lnPmtsum = lnPmt.sum(0)
            lnPonsum = lnPon.view(-1, 1).sum(0) 
            
            # negative multiply m
            lnPonsum = lnPonsum * self.negM
            loss = - (lnPmtsum + lnPonsum) /batchSize
            return loss
        else:
            neg_xx   =  x.narrow(1,1,batchSize-2)
            neg_x, _ = neg_xx.topk(self.negNum, dim=1, largest=True,sorted = False)
            
            pos_x = x.narrow(1,0,1)
            
            y = torch.cat((pos_x, neg_x),1)

            # denominator: sum of the exp weights
            Pmt_div = y.narrow(1,0,self.negNum+1).sum(1)

            # positive probability
            Pmt = y.select(1,0)
            lnPmt = torch.div(Pmt, Pmt_div)
            
            # negative probability
            Pon_div = Pmt_div.repeat(self.negNum,1)
            Pon = y.narrow(1,1,self.negNum)
            lnPon = torch.div(Pon, Pon_div.t())
            lnPon = -lnPon.add(-1)
            
            lnPmt.log_()
            lnPon.log_()
            
            lnPmtsum = lnPmt.sum(0)
            lnPonsum = lnPon.view(-1, 1).sum(0)
            
            # negative multiply m
            lnPonsum = lnPonsum * self.negM
            loss = - (lnPmtsum + lnPonsum) /(self.negNum +1)
            return loss

def GenerateIdx(batch_size,tmp=None):
    ''' Generate the idx matrix to make sure 
        The first column is positive, other columns are negative    
    Example:
        batch_size = 3*2
        
        idx =[ 3 1 2 5 4,
               4 0 2 3 5,
               5 1 0 3 4,
               1 0 2 3 5,
               2 1 0 3 4]
    '''   
    # # Method 3: Random select batchSize memory within batch
    idx = np.tile(np.arange(batch_size),batch_size).reshape(batch_size,batch_size) 
    for i in range(batch_size/2):
        idx[i,i] = 0
        idx[i,i+batch_size/2] = batch_size-1
        idx[i,0] = i + batch_size/2
        idx[i+batch_size/2,i] = 0
        idx[i+batch_size/2,0] = i
        idx[i+batch_size/2,i+batch_size/2] = batch_size-1
        
    return idx[:,:-1]