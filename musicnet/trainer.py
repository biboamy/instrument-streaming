import time
import torch.optim as optim
from lib import sp_loss
import sys
import torch

class Trainer:
    def __init__(self, model, lr, epoch, save_fn, avg, std):
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn
        #self.Xavg, self.Xstd = Variable(torch.from_numpy(avg).cuda()), Variable(torch.from_numpy(std).cuda())
    
    def fit(self, tr_loader, we, device):
        st = time.time()

        #save dict
        save_dict = {}
        save_dict['tr_loss'] = []

        print('Start Training #Epoch:%d'%(self.epoch))
        for e in range(1, self.epoch+1):
            #learning rate and optimizer
            lr = self.lr / (((e//(70*1))*2)+1) 
            loss_total = 0

            print( '\n==> Training Epoch #%d lr=%4f'%(e, lr))
            
            # Training
            for batch_idx, _input in enumerate(tr_loader):
                self.model.train()
                data, target = (_input[0].float().to(device)), \
                				[(_input[1].float().to(device)), (_input[2].float().to(device)), (_input[3].float().to(device))]
                #start feed in    
                predict = self.model(data)
                param1 = list(filter(lambda p: p.requires_grad, self.model.parameters()))
                opt1 = optim.SGD(param1, lr=lr, momentum=0.9, weight_decay=1e-4)
                opt1.zero_grad()    
    
                loss = sp_loss(predict, target, we, True, data)
                loss_total = sum(loss)
                loss_total.backward()
                opt1.step()
                    
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d] Inst %4f  pitch %4f  Stream %4f   Time %d'
                    %(e, self.epoch, batch_idx+1, len(tr_loader), loss[0].data, loss[1].data, loss[2].data, time.time() - st))
                sys.stdout.flush()
            print ('\n')
            save_dict['state_dict'] = self.model.state_dict()
            #save_dict['avg'] = self.Xavg
            #save_dict['std'] = self.Xstd
            torch.save(save_dict, self.save_fn+'e_%d'%(e))