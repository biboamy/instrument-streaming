import os
import torch
from lib import Data2Torch, model_init
from model import Net
from trainer import Trainer
import numpy as np

saveName = 'streaming'
batch_size = 16
lr = 0.01
epoch = 100

out_model_fn = './model/%s/'%(saveName)
if not os.path.exists(out_model_fn):
    os.makedirs(out_model_fn)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

t_kwargs = {'batch_size': batch_size, 'pin_memory': True}
tr_loader = torch.utils.data.DataLoader(Data2Torch(device, 'tr'), shuffle=True, **t_kwargs)
va_loader = torch.utils.data.DataLoader(Data2Torch(device, 'te'), shuffle=True, **t_kwargs)

model = Net().to(device)
model.apply(model_init)

mp = np.array([794532,230484,99407,99132,24426,14954,11468,8696,8310,4914,3006])
mmp = mp.astype(np.float32) / mp.sum()
cc=((mmp.mean() / mmp) * ((1-mmp)/(1 - mmp.mean())))**0.3
inverse_feq = torch.from_numpy(cc)
print(inverse_feq)

Trer = Trainer(model, lr, epoch, out_model_fn, 1, 1)
Trer.fit(tr_loader, inverse_feq, device)