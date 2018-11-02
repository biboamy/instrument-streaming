#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys, os, torch, librosa, matplotlib, datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
sys.path.append('./function')
from model import *
import numpy as np
from torch.autograd import Variable
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.init as init
from torch.utils.data import Dataset
date = datetime.datetime.now()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
def load_te_mp3(name,avg, std):
    def logCQT(file):
        sr = 16000
        y, sr = librosa.load(file,sr=sr)
        cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=27.5, n_bins=88, bins_per_octave=12)
        return ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)) + 1.0

    def chunk_data(f):
        s = int(16000*10/512)
        num = 88
        xdata = np.transpose(f)
        x = [] 
        xdata = xdata[:int(len(xdata)/s)*s,:]
        for i in range(int(len(xdata)/s)):
            data=xdata[i*s:i*s+s]
            x.append(data)
        return np.array(x)

    x = logCQT('mp3/'+name)
    x = chunk_data(x)
    x = np.transpose(x,(0, 2, 1))
    return x

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

# Dataset
class Data2Torch(Dataset):
    def __init__(self, data):
        self.X = data[0]

    def __getitem__(self, index):
        mX = torch.from_numpy(self.X[index]).float()
        return mX
    
    def __len__(self):
        return len(self.X)

def main(argv):
    #name = argv[0]
    name = argv[1]

    #load model dictionary
    save_dic = torch.load('data/model_100')

    #load model
    model = Net().cuda()
    model.apply(model_init)
    model_dict = model.state_dict()
    pretrained_dict1 = {k: v for k, v in save_dic['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict1) 
    model.load_state_dict(model_dict)
    print ('finishing loading model')

    #load test dataset
    Xavg, Xstd = save_dic['avg'], save_dic['std']
    Xte = load_te_mp3(name,Xavg.data.cpu().numpy(), Xstd.data.cpu().numpy())
    print ('finishing loading dataset')

    #predict configure
    v_kwargs = {'batch_size': 8, 'num_workers': 10, 'pin_memory': True}
    loader = torch.utils.data.DataLoader(Data2Torch([Xte]), **v_kwargs)

    s = Xte.shape
    pred_inst = np.zeros((s[0],10,s[2]))
    pred_pitch = np.zeros((s[0],88,s[2]))
    pred_roll = np.zeros((s[0],10,88,s[2]))
    #start predict
    print ('start predicting...')
    model.eval()
    ds = 0
    for idx,_input in enumerate(loader):
        data = Variable(_input.cuda())
        pred = model(data, Xavg, Xstd)
        pred_inst[ds: ds + len(data)] = np.repeat(F.sigmoid(pred[0]).data.cpu().numpy(), 2, axis=2)
        pred_pitch[ds: ds + len(data)] = np.repeat(F.sigmoid(pred[1]).data.cpu().numpy(), 2, axis=2)
        pred_roll[ds: ds + len(data)] = np.repeat(F.sigmoid(pred[2]).data.cpu().numpy(), 2, axis=3)
        ds += len(data)

    threshold = 0.85

    pred_inst = all_pred = np.transpose(pred_inst, (1, 0, 2)).reshape((10,-1))
    pred_pitch = np.transpose(pred_pitch, (1, 0, 2)).reshape((88,-1))
    pred_roll = np.transpose(pred_roll, (1, 2, 0, 3)).reshape((10,88,-1))

    pred_inst = np.delete(pred_inst,[3],axis=0)
    pred_roll = np.delete(pred_roll,[3],axis=0)

    pred_pitch[pred_pitch>threshold] = 1
    pred_pitch[pred_pitch<=threshold] = 0

    pred_roll[pred_roll>threshold] = 1
    pred_roll[pred_roll<=threshold] = 0

    np.save('output_data/inst/'+name[:-4]+'.npy',pred_inst)
    np.save('output_data/pitch/'+name[:-4]+'.npy',pred_pitch)
    np.save('output_data/roll/'+name[:-4]+'.npy',pred_roll)
    
if __name__ == "__main__":
    main(sys.argv)
    