import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from torch.autograd import Variable

# Dataset
class Data2Torch(Dataset):
    def __init__(self, data):
        self.X = data[0]
        self.Y = data[1]
        self.YP = data[2]
        self.YS = data[3]

    def __getitem__(self, index):

        mX = torch.from_numpy(self.X[index]).float()
        mY = torch.from_numpy(self.Y[index]).float()
        mYP = torch.from_numpy(self.YP[index]).float()
        mYS = torch.from_numpy(self.YS[index]).float()
        return mX, mY, mYP, mYS#, mYstft#, mXM
    
    def __len__(self):
        return len(self.X)

# lib
def sp_loss(pred, tar, gwe, isPitchLoss, xgt=None):

    we = gwe[0].cuda()
    wwe = 10
    we *= wwe
    
    loss = 0

    def inst_loss(inst_pred, inst_tar):
        loss_i = 0
        for idx, (out, fl_target) in enumerate(zip(inst_pred,inst_tar)):
            twe = we.view(-1,1).repeat(1,fl_target.size(1)).type(torch.cuda.FloatTensor)
            ttwe = twe * fl_target.data + (1 - fl_target.data) * 1
            loss_fn = nn.BCEWithLogitsLoss(weight=ttwe, size_average=True)
            loss_i += loss_fn(torch.squeeze(out), fl_target)
        
        return loss_i

    def pitch_loss(pit_pred, pit_tar):
        loss_p = 0
        for idx, (out, fl_target) in enumerate(zip(pit_pred,pit_tar)):
            ttwe = 10 * fl_target.data + (1 - fl_target.data) * 1
            loss_fn = nn.BCEWithLogitsLoss(weight=ttwe, size_average=True)
            loss_p += loss_fn(out, fl_target)
        return loss_p

    def stream_loss(str_pred, str_tar):

        loss_s = 0
        for idx, (out, fl_target) in enumerate(zip(str_pred,str_tar)):
            ttwe = 10 * fl_target.data + (1 - fl_target.data) * 1
            loss_fn = nn.BCEWithLogitsLoss(weight=ttwe, size_average=True)
            los = loss_fn(out, fl_target)
            loss_s += los
         
        return loss_s

    def re_loss(pred, tar):
        loss_r = 0
        for idx, (out, fl_target) in enumerate(zip(pred,tar)):
            loss_fn = nn.MSELoss(size_average=True)
            loss_r += loss_fn(out, fl_target)
        return loss_r

    def des_loss(pred, tar):
        loss_d = 0
        loss_fn = nn.BCEWithLogitsLoss(size_average=True)
        for idx, (out, fl_target) in enumerate(zip(pred,tar)):
            loss_d += loss_fn(out, fl_target)
        return loss_d
   
    l1 = inst_loss(pred[0],tar[0])
    l2 = pitch_loss(pred[1],tar[1])
    l3 = stream_loss(pred[2],tar[2])*2
    return l1,l2,l3
       
def model_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
       # init.constant(m.bias, 0)

def num_params(model):
    params = 0
    for i in model.parameters():
        params += i.view(-1).size()[0]
    print( '#params:%d'%(params))

