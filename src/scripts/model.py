import os
import numpy as np
import torch
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class Model:
    # TODO various metrics
    
    def __init__(self, model, criterion, optimizer, model_path=''):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.epoch = 0
        self.best_loss = np.inf
        self.model_path = model_path
        
    
    def _run_epoch(self, data_loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        loss_meter = AverageMeter()

        for i, (input, target) in enumerate(data_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            if train:
                self.optimizer.zero_grad()

            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            if train:
                loss.backward()
                self.optimizer.step()

            loss_meter.update(loss.data[0], input.size()[0])

        return loss_meter.avg
    
    
    def validate(self, val_loader):
        return self._run_epoch(val_loader, train=False)
    
    
    def fit(self, train_loader, val_loader, n_epoch=1):
        history = {stat:[] for stat in ['train', 'val']}
        val_loss = self.validate(val_loader)
        print("Val: {0}".format(val_loss)) 
        for _ in range(n_epoch):
            train_loss = self._run_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_model(self.model_path)
            
            print("Epoch {0}: train {1} \t val {2}".format(self.epoch, train_loss, val_loss))
            self.epoch += 1
            
            history['train'].append(train_loss)
            history['val'].append(val_loss)
        return history
            
    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def save_model(self, filename):
        if filename:
            state = {
                'state_dict' : self.model.state_dict(),
                 # 'optimizer' : self.optimizer.state_dict(),
                'epoch' : self.epoch,
                'best_loss' : self.best_loss
            }
            torch.save(state, filename)
            print("Model saved")
        
    def load_model(self, filename):
        if os.path.isfile(filename):
            state = torch.load(filename)
            self.model.load_state_dict(state['state_dict'])
            # self.optimizer.load_state_dict(state['optimizer'])
            self.epoch = state['epoch']
            self.best_loss = state['best_loss']
        else:
            raise Exception("No state found at {}".format(filename))
            
    def predict_proba(self, images):
        softmax = nn.Softmax()
        scores = self.model(images)
        return softmax(scores)