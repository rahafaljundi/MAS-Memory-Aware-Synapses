
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import sys


import shutil
import pdb

import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs



class MNIST_Split(datasets.MNIST):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
  
    

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,digits=[1,2]):
        super(MNIST_Split, self).__init__(root, train, transform, target_transform, download)



        #get only the two digits
        self.digit_labels=None
        self.digit_data=None
        self.classes= digits 
        if self.train:
            
            #loop over the given digits and extract there corresponding data
            for digit in digits:
                digit_mask=torch.eq(self.train_labels , digit) 
                digit_index=torch.nonzero(digit_mask)
                digit_index=digit_index.view(-1)
                this_digit_data=self.train_data[digit_index]
                this_digit_labels=self.train_labels[digit_mask]
                this_digit_labels.fill_(digits.index(digit))
                if self.digit_data is None:
                    self.digit_data=this_digit_data.clone()
                    self.digit_labels=this_digit_labels.clone()
                else:
                    self.digit_data=torch.cat((self.digit_data,this_digit_data),0)
                    self.digit_labels=torch.cat((self.digit_labels,this_digit_labels),0)
            #self.train_data, self.train_labels = torch.load(
                #os.path.join(root, self.processed_folder, self.training_file))
        else:
                       #loop over the given digits and extract there corresponding data
            for digit in digits:
                digit_mask=torch.eq(self.test_labels , digit) 
                digit_index=torch.nonzero(digit_mask)
                digit_index=digit_index.view(-1)
                this_digit_data=self.test_data[digit_index]
                this_digit_labels=self.test_labels[digit_mask]
                this_digit_labels.fill_(digits.index(digit))
                if self.digit_data is None:
                    self.digit_data=this_digit_data.clone()
                    self.digit_labels=this_digit_labels.clone()
                    
                else:
                    self.digit_data=torch.cat((self.digit_data,this_digit_data),0)
                    self.digit_labels=torch.cat((self.digit_labels,this_digit_labels),0)
                    
         
        
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
       
        img, target = self.digit_data[index], self.digit_labels[index]
       

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img=img.view(-1,28*28) 
       
        return img, target

    def __len__(self):
        return(self.digit_labels.size(0)) 

  