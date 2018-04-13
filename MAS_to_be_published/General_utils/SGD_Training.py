
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import copy
import os
import pdb
import shutil
from torch.utils.data import DataLoader
#end of imports

def train_model(model, criterion, optimizer, lr_scheduler,lr,dset_loaders,dset_sizes,use_gpu, num_epochs,exp_dir='./',resume=''):
    print('dictoinary length'+str(len(dset_loaders)))
    #reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        #model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        #modelx = checkpoint['model']
        #model.reg_params=modelx.reg_params
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        #pdb.
        #model.reg_params=reg_params
        #del model.reg_params
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
            start_epoch=0
            print("=> no checkpoint found at '{}'".format(resume))
    
    print(str(start_epoch))
    #pdb.set_trace()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch,lr)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs=inputs.squeeze()
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                model.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    #print('step')
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                del outputs
                del labels
                del inputs
                del loss
                del preds
                best_acc = epoch_acc
                #best_model = copy.deepcopy(model)
                torch.save(model,os.path.join(exp_dir, 'best_model.pth.tar'))
                
        #epoch_file_name=exp_dir+'/'+'epoch-'+str(epoch)+'.pth.tar'
        epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'epoch_acc':epoch_acc,
            'arch': 'alexnet',
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
                },epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    #best_model = copy.deepcopy(model)
    torch.save(state, filename)
   