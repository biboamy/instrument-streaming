import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from structure import *
batchNorm_momentum = 0.1

class block(nn.Module):
    def __init__(self, inp, out, ksize, pad, ds_ksize, ds_stride):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, padding=pad)
        self.bn1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.conv2 = nn.Conv2d(out, out, kernel_size=ksize, padding=pad)
        self.bn2 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.skip = nn.Conv2d(inp, out, kernel_size=1, padding=0)
        self.ds = nn.Conv2d(out, out, kernel_size=ds_ksize, stride=ds_stride, padding=0)

    def forward(self, x):
        x11 = F.leaky_relu(self.bn1(self.conv1(x)))
        x12 = F.leaky_relu(self.bn2(self.conv2(x11)))
        x12 += self.skip(x)
        xp = self.ds(x12)
        return xp, xp, x12.size()

class d_block(nn.Module):
    def __init__(self, inp, out, isLast, ksize, pad, ds_ksize, ds_stride):
        super(d_block, self).__init__()
        self.conv2d = nn.ConvTranspose2d(inp, int(inp/2), kernel_size=ksize, padding=pad)
        self.bn2d = nn.BatchNorm2d(int(inp/2), momentum= batchNorm_momentum)
        self.conv1d = nn.ConvTranspose2d(int(inp/2), out, kernel_size=ksize, padding=pad)
        if not isLast: 
            self.bn1d = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
            self.us = nn.ConvTranspose2d(inp-out, inp-out, kernel_size=ds_ksize, stride=ds_stride)
        else: self.us = nn.ConvTranspose2d(inp, inp, kernel_size=2, stride=(2,2))

    def forward(self, x, idx, size, isLast, skip):
        x = self.us(x,output_size=size)
        if not isLast: x = torch.cat((x, skip), 1)
        x = F.leaky_relu(self.bn2d(self.conv2d(x)))
        if isLast: x = self.conv1d(x)
        else:  x = F.leaky_relu(self.bn1d(self.conv1d(x)))
        return x

class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()

        self.bn = nn.BatchNorm2d(1, momentum= batchNorm_momentum)

        self.block1 = block(ENCODER['b1']['inp'],ENCODER['b1']['oup'],ENCODER['b1']['k'],ENCODER['b1']['p'],ENCODER['b1']['ds_k'],ENCODER['b1']['ds_s'])
        self.block2 = block(ENCODER['b2']['inp'],ENCODER['b2']['oup'],ENCODER['b2']['k'],ENCODER['b2']['p'],ENCODER['b2']['ds_k'],ENCODER['b2']['ds_s'])
        self.block3 = block(ENCODER['b3']['inp'],ENCODER['b3']['oup'],ENCODER['b3']['k'],ENCODER['b3']['p'],ENCODER['b3']['ds_k'],ENCODER['b3']['ds_s'])
        self.block4 = block(ENCODER['b4']['inp'],ENCODER['b4']['oup'],ENCODER['b4']['k'],ENCODER['b4']['p'],ENCODER['b4']['ds_k'],ENCODER['b4']['ds_s'])

        self.conv1 = nn.Conv2d(ENCODER['s1']['inp'],ENCODER['s1']['oup'], kernel_size=ENCODER['s1']['k'], padding=ENCODER['s1']['p'])
        self.conv2 = nn.Conv2d(ENCODER['s2']['inp'],ENCODER['s2']['oup'], kernel_size=ENCODER['s2']['k'], padding=ENCODER['s2']['p'])
        self.conv3 = nn.Conv2d(ENCODER['s3']['inp'],ENCODER['s3']['oup'], kernel_size=ENCODER['s3']['k'], padding=ENCODER['s3']['p'])

    def forward(self, x):

        x = self.bn(x)
       
        x1,idx1,s1 = self.block1(x)
        x2,idx2,s2 = self.block2(x1)
        x3,idx3,s3 = self.block3(x2)
        x4,idx4,s4 = self.block4(x3)
    
        c1=self.conv1(x3)
        c2=self.conv2(x2)
        c3=self.conv3(x1)

        return x4,[idx1,idx2,idx3,idx4],[s1,s2,s3,s4],[c1,c2,c3,x1]


class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()
        self.d_block1 = d_block(DECODER['db1']['inp'],DECODER['db1']['oup'],False,DECODER['db1']['k'],DECODER['db1']['p'],DECODER['db1']['ds_k'],DECODER['db1']['ds_s'])
        self.d_block2 = d_block(DECODER['db2']['inp'],DECODER['db2']['oup'],False,DECODER['db2']['k'],DECODER['db2']['p'],DECODER['db2']['ds_k'],DECODER['db2']['ds_s'])
        self.d_block3 = d_block(DECODER['db3']['inp'],DECODER['db3']['oup'],False,DECODER['db3']['k'],DECODER['db3']['p'],DECODER['db3']['ds_k'],DECODER['db3']['ds_s'])
        self.d_block4 = d_block(DECODER['db4']['inp'],DECODER['db4']['oup'],True,DECODER['db4']['k'],DECODER['db4']['p'],DECODER['db4']['ds_k'],DECODER['db4']['ds_s'])

    def forward(self, x, idx, s, c):
        x = self.d_block1(x,idx[3],s[3],False,c[0])
        x = self.d_block2(x,idx[2],s[2],False,c[1])
        x = self.d_block3(x,idx[1],s[1],False,c[2])
        pred = self.d_block4(x,idx[0],s[0],True,c[3])
       
        return pred
