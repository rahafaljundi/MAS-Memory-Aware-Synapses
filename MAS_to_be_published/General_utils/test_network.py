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
from ImageFolderTrainVal import *

import pdb
def test_model(model_path,dataset_path,batch_size=100):
    model=torch.load(model_path)
    model.eval()
    model=model.cuda()
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x],batch_size ,
                                                   shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    class_correct = list(0. for i in range(len(dset_classes)))
    class_total = list(0. for i in range(len(dset_classes)))
    for data in dset_loaders['val']:
        images, labels = data
        images=images.cuda()
        images=images.squeeze()
        labels=labels.cuda()
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        #pdb.set_trace()
        for i in range(len(predicted)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
        del images
        del labels
        del outputs
        del data
    if 0:
        for i in range(len(dset_classes)):
            print('Accuracy of %5s : %2d %%' % (
            dset_classes[i], 100 * class_correct[i] / class_total[i]))
    accuracy=np.sum(class_correct)*100/np.sum(class_total)
    print('Accuracy: ' +str(accuracy))
    return accuracy

def test_model_lwtaDeactive(model_path,dataset_path,batch_size=100):
    model=torch.load(model_path)
    model=remove_lwta(model )
    model=model.cuda()
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x],batch_size ,
                                                   shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    class_correct = list(0. for i in range(len(dset_classes)))
    class_total = list(0. for i in range(len(dset_classes)))
    for data in dset_loaders['val']:
        images, labels = data
        images=images.cuda()
        labels=labels.cuda()
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        #pdb.set_trace()
        for i in range(len(predicted)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
        del images
        del labels
        del outputs
        del data
    if 0:
        for i in range(len(dset_classes)):
            print('Accuracy of %5s : %2d %%' % (
            dset_classes[i], 100 * class_correct[i] / class_total[i]))
    accuracy=np.sum(class_correct)*100/np.sum(class_total)
    print('Accuracy: ' +str(accuracy))
    return accuracy

def test_model_sparce(model_path,dataset_path,batch_size=100):
    model=torch.load(model_path)
    
    model=model.cuda()
    
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x],batch_size ,
                                                   shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    class_correct = list(0. for i in range(len(dset_classes)))
    class_total = list(0. for i in range(len(dset_classes)))
    for data in dset_loaders['val']:
        images, labels = data
        images=images.cuda()
        images=images.squeeze()
        labels=labels.cuda()
        outputs,x = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        #pdb.set_trace()
        for i in range(len(predicted)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
        del images
        del labels
        del outputs
        del data
    if 0:
        for i in range(len(dset_classes)):
            print('Accuracy of %5s : %2d %%' % (
            dset_classes[i], 100 * class_correct[i] / class_total[i]))
    accuracy=np.sum(class_correct)*100/np.sum(class_total)
    print('Accuracy: ' +str(accuracy))
    return accuracy



def remove_lwta(LWTA_model ):
    
 
    new_model=models.alexnet(pretrained=True)
    new_model.features=LWTA_model.features
    new_model.classifier = nn.Sequential()
    new_model.classifier.add_module('0', LWTA_model.classifier._modules['0'])
    new_model.classifier.add_module('1',  LWTA_model.classifier._modules['1'])
    new_model.classifier.add_module('2',  LWTA_model.classifier._modules['2'])
    new_model.classifier.add_module('3', LWTA_model.classifier._modules['3'])
    #LWTA_model.classifier.add_module('lwt1', LWTA(4096,4096,window_size))
    new_model.classifier.add_module('4',  LWTA_model.classifier._modules['4'])
    new_model.classifier.add_module('5',  LWTA_model.classifier._modules['5'])
    #LWTA_model.classifier.add_module('lwt2', LWTA(4096,4096,window_size))
    new_model.classifier.add_module('6', LWTA_model.classifier._modules['6'])
    return new_model


def test_model_animals(model_path,dataset_path,batch_size=4,correspond=[280, 291, 292, 293, 296]):
    
    model=torch.load(model_path)
    
    model=model.cuda()
    model.eval()
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x],batch_size ,
                                                   shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    class_correct = list(0. for i in range(len(dset_classes)))
    class_total = list(0. for i in range(len(dset_classes)))
    for data in dset_loaders['val']:
        images, labels = data
        images=images.cuda()
        labels=labels.cuda()
       
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
       
    
        for i in range(len(predicted)):
            label = labels[i]
            if correspond[label]==predicted[i]:
                class_correct[label] += 1
            class_total[label] += 1
        del images
        del labels
        del outputs
        del data
    if 1:
        for i in range(len(dset_classes)):
            print('Accuracy of %5s : %2d %%' % (
            dset_classes[i], 100 * class_correct[i] / class_total[i]))
    accuracy=np.sum(class_correct)*100/np.sum(class_total)
    print('Accuracy: ' +str(accuracy))
    return accuracy