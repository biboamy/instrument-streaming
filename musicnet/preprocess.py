import numpy as np
import librosa
import h5py
import math
import matplotlib.pyplot as plt
import librosa

test_list = ['2303', '2191', '2382', '2628', '2416', '2556', '2298', '1819', '1759', '2106']

def extract_cqt():
	def logCQT(y, sr):
		cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=27.5, n_bins=88, bins_per_octave=12)
		return ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)) + 1.0
	def chunk(data, chunk_size):
		chunk_length = int(np.ceil(data.shape[-1]/chunk_size))
		oup = np.zeros((chunk_length, data.shape[0], chunk_size))
		for j in range(chunk_length):
			inp = data[:,j*chunk_size:(j+1)*chunk_size]
			oup[j,:,:inp.shape[-1]] = inp
		return oup

	train_data = np.load('./data/musicnet.npz', encoding='bytes')
	chunk_size = int(6*44100/512)
	index = 0

	#hf_tr = h5py.File('./data/tr_CQT.h5', 'a')
	#hf_te = h5py.File('./data/te_CQT.h5', 'a')
	#hf_tr.create_dataset('cqt', shape=(20433, 88, chunk_size), chunks=(1, 88, chunk_size), maxshape=(None, 88, chunk_size))
	#hf_te.create_dataset('cqt', shape=(252, 88, chunk_size), chunks=(1, 88, chunk_size), maxshape=(None, 88, chunk_size))

	for key in test_list:#train_data.keys():
		x = train_data[key][0][:]
		
		index+=(np.ceil(len(x)/512/chunk_size))

		feature = logCQT(x, 44100)
		feature = chunk(feature, chunk_size)
		print(key, feature.shape, index)
		if key in test_list:
			hf_te["cqt"][int(index):int(index+feature.shape[0])] = feature
		else:
			hf_tr["cqt"][int(index):int(index+feature.shape[0])] = feature
		
		index+=feature.shape[0]
		
	print(index)
	
#extract_cqt()
def extract_label():
	def chunk(data, chunk_size):
		chunk_length = int(np.ceil(data.shape[-1]/chunk_size))
		oup = np.zeros((chunk_length, data.shape[0], chunk_size))
		for j in range(chunk_length):
			inp = data[:,j*chunk_size:(j+1)*chunk_size]
			oup[j,:,:inp.shape[-1]] = inp
		return oup

	inst_lookup = {1:0,41:1,42:2,43:3,72:4,71:5,61:6,69:7,74:8,7:9,44:10}
	data = np.load('./data/musicnet.npz', encoding='bytes')
	chunk_size = int(6*44100/512)
	stride = 512
	index = 0
	
	hf_tr = h5py.File('./data/tr_labels.h5', 'a')
	hf_te = h5py.File('./data/te_labels.h5', 'a')
	hf_tr.create_dataset('inst', shape=(20433, 11, chunk_size), chunks=(1, 11, chunk_size), maxshape=(None, 11, chunk_size))
	hf_tr.create_dataset('pitch', shape=(20433, 128, chunk_size), chunks=(1, 128, chunk_size), maxshape=(None, 128, chunk_size))
	hf_tr.create_dataset('stream', shape=(20433, 11, 128, chunk_size), chunks=(1, 11, 128, chunk_size), maxshape=(None, 11, 128, chunk_size))
	hf_te.create_dataset('inst', shape=(252, 11, chunk_size), chunks=(1, 11, chunk_size), maxshape=(None, 11, chunk_size))
	hf_te.create_dataset('pitch', shape=(252, 128, chunk_size), chunks=(1, 128, chunk_size), maxshape=(None, 128, chunk_size))
	hf_te.create_dataset('stream', shape=(252, 11, 128, chunk_size), chunks=(1, 11, 128, chunk_size), maxshape=(None, 11, 128, chunk_size))

	for i,k in enumerate(data.keys()):
		x, y = data[k]
		length=int(np.ceil(len(x)/512/chunk_size))
		Ys = np.zeros((length,11, 128, chunk_size))
		for l in range(length):
			for c in range(chunk_size):
				window = chunk_size*l + c
				labels = y[window*stride]
				for label in labels:
					Ys[l,inst_lookup[label.data[0]],label.data[1],c] = 1
		print(i,k,Ys.shape)
		
		Yi = Ys.sum(2)
		Yi[Yi>0]=1
		Yp = Ys.sum(1)
		Yp[Yp>0]=1
		if k in test_list:
			hf_te["inst"][int(index):int(index+Yi.shape[0])] = Yi
			hf_te["pitch"][int(index):int(index+Yp.shape[0])] = Yp
			hf_te["stream"][int(index):int(index+Ys.shape[0])] = Ys
		else:
			hf_tr["inst"][int(index):int(index+Yi.shape[0])] = Yi
			hf_tr["pitch"][int(index):int(index+Yp.shape[0])] = Yp
			hf_tr["stream"][int(index):int(index+Ys.shape[0])] = Ys
		index+=Yi.shape[0]

#extract_label()
def visualize():
	cqt = h5py.File('./data/musicnetCQT.h5', 'r')
	label = h5py.File('./data/musicnetLabel.h5', 'r')
	for k in cqt.keys():
		print(cqt[k].shape, label[k.split('_')[1]][:].sum(-1).sum(-1))
		
		plt.plot()
		plt.imshow(cqt[k][:,:], aspect='auto')
		plt.savefig('x.png')
		plt.plot()
		plt.imshow(label[k.split('_')[1]][:,21:109,:800].sum(0), aspect='auto')
		plt.savefig('y1.png')
		plt.plot()
		plt.imshow(label[k.split('_')[1]][2,:,:800], aspect='auto')
		plt.savefig('y2.png')
		plt.imshow(label[k.split('_')[1]][3,:,:800], aspect='auto')
		plt.savefig('y3.png')
		print('end')
		print(hi)
		
#visualize()

def get_weight(Ytr):
	def _get_weight(Ytr):
		mp = Ytr[:].sum(0).sum(1)
		mmp = mp.astype(np.float32) / mp.sum()
		cc=((mmp.mean() / mmp) * ((1-mmp)/(1 - mmp.mean())))**0.3
		cc[3]=1
		inverse_feq = torch.from_numpy(cc)
		return inverse_feq

	inst_lookup = {1:0,41:1,42:2,43:3,72:4,71:5,61:6,69:7,74:8,7:9,44:10}
	train_data = h5py.File('./data/musicnet.h5', 'r')
	#train_data = np.load('./data/musicnet.npz', encoding='bytes')
	for key in train_data.keys():
		x = train_data[key]['data']
		
		Yvec = np.zeros((11, int(len(x)/512)))
		for window in range(Yvec.shape[1]):
		    labels = Y[window*stride]
		    for label in labels:
		        Yvec[inst_lookup[label.data[0]],window] = 1


	