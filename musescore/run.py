import torch.optim as optim
import datetime
date = datetime.datetime.now()
import sys, os
sys.path.append('./function')
from lib import *
from fit import *
from model import *
from loadData import *
os.environ['CUDA_VISIBLE_DEVICES'] = '2' # change

saveName = 'streaming'
batch_size = 8

def get_weight(Ytr):
	mp = Ytr[:].sum(0).sum(1)
	mmp = mp.astype(np.float32) / mp.sum()
	cc=((mmp.mean() / mmp) * ((1-mmp)/(1 - mmp.mean())))**0.3
	cc[3]=1
	inverse_feq = torch.from_numpy(cc)
	return inverse_feq

out_model_fn = './data/model/%d%d%d/%s/'%(date.year,date.month,date.day,saveName)
if not os.path.exists(out_model_fn):
    os.makedirs(out_model_fn)

# load data
t_kwargs = {'batch_size': batch_size, 'num_workers': 2, 'pin_memory': True,'drop_last': True}
v_kwargs = {'batch_size': batch_size, 'num_workers': 2, 'pin_memory': True}
Xtr,Ytr,avg,std = load()

Ytr_p = load_pitch()
Ytr_s = load_stream()
trdata = [Xtr[:], Ytr[:], Ytr_p, Ytr_s]
tr_loader = torch.utils.data.DataLoader(Data2Torch(trdata), shuffle=True, **t_kwargs)
print('finishing data building...')

# build model
model = Net().cuda()
model.apply(model_init)

inverse_feq = get_weight(Ytr)

# start training
Trer = Trainer(model, 0.01, 100, out_model_fn, avg, std)
Trer.fit(tr_loader, inverse_feq)

print( out_model_fn)