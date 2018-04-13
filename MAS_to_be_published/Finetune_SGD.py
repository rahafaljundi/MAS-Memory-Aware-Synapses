from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as pltD
import time
import copy
import os
import shutil
import sys
sys.path.append('General_utils')

from ImageFolderTrainVal import *
from test_network import *
from SGD_Training import *

import pdb

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    print('lr is '+str(lr))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
def fine_tune_SGD(dataset_path,model_path,exp_dir,batch_size=100, num_epochs=100,lr=0.0004,init_freeze=1):
   
    print('lr is ' + str(lr))
    
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            model_ft = checkpoint['model']
    if not os.path.isfile(model_path):
        model_ft = models.alexnet(pretrained=True)
       
    else:
        model_ft=torch.load(model_path)
    if not init_freeze:    
        num_ftrs = model_ft.classifier[6].in_features 
        model_ft.classifier._modules['6'] = nn.Linear(num_ftrs, len(dset_classes))    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()


    
    optimizer_ft =  optim.SGD(model_ft.parameters(), lr, momentum=0.9)

        
    
  
    model_ft = train_model(model_ft, criterion, optimizer_ft,exp_lr_scheduler,lr, dset_loaders,dset_sizes,use_gpu,num_epochs,exp_dir,resume)
    
    return model_ft
