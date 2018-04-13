
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
sys.path.append('../General_utils')
sys.path.append('../MAS_utils')#temporary
from ImageFolderTrainVal import *
from test_network import *
import warnings
warnings.filterwarnings('ignore')
import pdb


def test_seq_task_performance(previous_model_path,current_model_path,dataset_path,check=0):
    if check:
        check_current_model_ft=torch.load(current_model_path)
        current_model_ft=check_current_model_ft['model']
    else:
         current_model_ft=torch.load(current_model_path)
        
    previous_model_ft=torch.load(previous_model_path)
       
    last_layer_index=str(len(previous_model_ft.classifier._modules)-1)
    
    current_model_ft.classifier._modules[last_layer_index] = previous_model_ft.classifier._modules[last_layer_index]
    
    #pdb.set_trace()
    temp_path='tobetested.pth.tar'
    torch.save(current_model_ft,temp_path)
    acc=test_model(temp_path,dataset_path)
    return acc




def compute_each_step_forgetting(models_path,datasets_path):
    
    res=[]
    avgs=[]
    for i in range(len(datasets_path)-1,len(datasets_path)):

        for t in range(0,i+1):
            this_avg=0  
            current_model_path=models_path[i]
            previous_model_path=models_path[t]
            dataset_path=datasets_path[t]
            res_new=test_seq_task_performance_sparce(previous_model_path,current_model_path,dataset_path,check=0)

            if len(res)<=t:
                res.append(test_model_sparce(previous_model_path,dataset_path))
            this_avg=(res[t]-res_new)
        #this_avg=this_avg/i
            print(this_avg)
            avgs.append(this_avg)
    return res,avgs        