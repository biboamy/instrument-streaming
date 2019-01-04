#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import SharedArray as sa
import sys,h5py
import multiprocessing
sys.path.append('../')

data_name = 'musescore500'

def load():
    avg, std = np.load('data/cqt_avg_std.npy')
    try:
        Xtr = sa.attach('shm://%s_Xtr'%(data_name))
        Ytr = sa.attach('shm://%s_Ytr'%(data_name))
    except:
        vadata = h5py.File('ex_data/'+data_name+'/va.h5', 'r')
        trdata = h5py.File('ex_data/'+data_name+'/tr.h5', 'r')
        Xtr = sa.create('shm://%s_Xtr'%(data_name), (trdata['x'].shape), dtype='float32')
        Xtr[:] = trdata['x'][:]
        Ytr = sa.create('shm://%s_Ytr'%(data_name), (trdata['y'].shape), dtype='float32')
 
    return Xtr, Ytr, avg, std 

def load_pitch():
    try:
        Ytr_p = sa.attach('shm://%s_Ytr_pitch'%(data_name))
    except:
        vadata = h5py.File('ex_data/'+data_name+'/va_pitch.h5', 'r')
        trdata = h5py.File('ex_data/'+data_name+'/tr_pitch.h5', 'r')
        Ytr_p = sa.create('shm://%s_Ytr_pitch'%(data_name), (trdata['y'].shape), dtype='float32')
        Ytr_p[:] = trdata['y'][:]

    return Ytr_p

def load_stream():
   
    try:
        Ytr_s = sa.attach('shm://%s_Ytr_stream'%(data_name))
    except:
        vadata = h5py.File('ex_data/'+data_name+'/va_stream.h5', 'r')
        trdata = h5py.File('ex_data/'+data_name+'/tr_stream.h5', 'r')
        Ytr_s = sa.create('shm://%s_Ytr_stream'%(data_name), (trdata['y'].shape), dtype='float32')
       
    return Ytr_s


