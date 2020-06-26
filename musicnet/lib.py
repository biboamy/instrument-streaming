from torch.utils.data import Dataset
import torch
import h5py
import random
import numpy as np
import torch.nn.init as init
import librosa
import sys, math
import torch.nn as nn
import torch.nn.functional as F

class Data2Torch(Dataset):
	def __init__(self, device, choose):
		self.cqt_data = h5py.File('./data/'+choose+'_CQT.h5', 'r')
		self.label_data = h5py.File('./data/'+choose+'_labels.h5', 'r')
		self.seq_duration = 312
		self.device = device

	def __getitem__(self, index):
		x = self.cqt_data['cqt'][index]
		Yi = self.label_data['inst'][index]
		Yp = self.label_data['pitch'][index][21:109]
		Ys = self.label_data['stream'][index][:,21:109]
		
		return torch.from_numpy(x), torch.from_numpy(Yi), torch.from_numpy(Yp), torch.from_numpy(Ys)
	
	def __len__(self):
		return len(self.cqt_data['cqt'])

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

	l1 = inst_loss(F.max_pool1d(pred[0], 86),F.max_pool1d(tar[0], 86))
	l2 = pitch_loss(F.max_pool1d(pred[1], 86),F.max_pool1d(tar[1], 86))
	l3 = stream_loss(F.max_pool2d(pred[2], (1,86)),F.max_pool2d(tar[2],(1, 86)))*2
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