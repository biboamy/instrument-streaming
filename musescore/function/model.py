import torch.nn as nn
import torch, sys, math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
sys.path.append('../fun')
from block import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.numl = 9
        self.encode = Encode()
        self.decode = Decode()

    def forward(self, _input, Xavg, Xstd):
        
        def get_inst_x(x,avg,std):
            xs = x.size()
            avg = avg.view(1, avg.size()[0],1,1).repeat(xs[0], 1, xs[2], 1).type('torch.FloatTensor')
            std = std.view(1, std.size()[0],1,1).repeat(xs[0], 1, xs[2], 1).type('torch.FloatTensor')
            x = (x - avg)/std
            return x

        # data prepare
        x = _input.unsqueeze(3) 
        x = get_inst_x(x,Xavg,Xstd)
        x = x.permute(0,3,1,2)
        # encode timbre
        fea_vec,i,s,c = self.encode(x)
        # decode to piano roll
        pred = self.decode(fea_vec,i,s,c)
        pred = F.max_pool2d(pred,(1,2),(1,2)).squeeze(1)

        inst = torch.sum(pred,2)*0.01#torch.sum(pred,2)/pred.size()[2]
        pitch = torch.sum(pred,1)*0.1#torch.sum(pred,1)/pred.size()[1]
        inst_scale = F.sigmoid(inst.view(inst.size()[0],inst.size()[1],1,inst.size()[2]))

        predict = [inst, pitch, pred*inst_scale]
                
        return predict
        

        
        