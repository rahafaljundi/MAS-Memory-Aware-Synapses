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
sys.path.append('MAS_utils')
from ImageFolderTrainVal import *

from MAS_based_Training import *

from test_network import *
import pdb

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0004, lr_decay_epoch=54):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
   
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
def MAS(dataset_path,previous_task_model_path,exp_dir,data_dir,reg_sets,reg_lambda=1,norm='L2', num_epochs=100,lr=0.0008,batch_size=200,b1=False):
    """Call MAS on mainly a sequence of two object recognition tasks with a head for each 
    dataset_path=new dataset path
    exp_dir=where to save the new model
    previous_task_model_path: previous task in the sequence model path to start from 
    reg_sets,data_dir: sets of examples used to compute omega. Here the default is the training set of the last task
    b1=to mimic online importance weight computation, batch size=1
    reg_lambda= regulizer hyper parameter. In object recognition it was set to 1.
    norm=the norm used to compute the gradient of the learned function
    """
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=150,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    model_ft = torch.load(previous_task_model_path)
    use_gpu = torch.cuda.is_available()

    #update omega value
    if b1:
        update_batch_size=1
    else:
        update_batch_size=batch_size
    model_ft=update_weights_params(data_dir,reg_sets,model_ft,update_batch_size,norm)
        
    #set the lambda value for the MAS    
    model_ft.reg_params['lambda']=reg_lambda
    
    #get the number of features in this network and add a new task head
    last_layer_index=str(len(model_ft.classifier._modules)-1)

    num_ftrs=model_ft.classifier._modules[last_layer_index].in_features 
    model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))    
   
    #check the computed omega
    sanitycheck(model_ft)
    
    #define the loss function
    criterion = nn.CrossEntropyLoss()
   
    
    if use_gpu:
        model_ft = model_ft.cuda()
    
 
    #call the MAS optimizer
    optimizer_ft =Weight_Regularized_SGD(model_ft.parameters(), lr, momentum=0.9)
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    #train the model
    #this training functin passes the reg params to the optimizer to be used for penalizing changes on important params
    model_ft =train_model(model_ft, criterion, optimizer_ft,exp_lr_scheduler, lr,dset_loaders,dset_sizes,use_gpu,num_epochs,exp_dir,resume)
    
    return model_ft
def MAS_sequence(dataset_path,pevious_pathes,previous_task_model_path,exp_dir,data_dirs,reg_sets,reg_lambda=1,norm='L2', num_epochs=100,lr=0.0008,batch_size=200,weight_decay=1e-5,b1=False,after_freeze=1):
    """Call MAS on mainly a sequence of tasks with a head for each where between at each step it sees samples from all the previous tasks to approximate the importance weights 
    dataset_path=new dataset path
    exp_dir=where to save the new model
    previous_task_model_path: previous task in the sequence model path to start from 
    pevious_pathes:pathes of previous methods to use the previous heads in the importance weights computation. We assume that each task head is not changed in classification setup of different tasks
    reg_sets,data_dirs: sets of examples used to compute omega. Here the default is the training set of the last task
    b1=to mimic online importance weight computation, batch size=1
    reg_lambda= regulizer hyper parameter. In object recognition it was set to 1.
    norm=the norm used to compute the gradient of the learned function
    """
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=150,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
   
    model_ft = torch.load(previous_task_model_path)
    
    if b1:
        update_batch_size=1
    else:
        update_batch_size=batch_size
    model_ft=update_sequence_MAS_weights(data_dirs,reg_sets,pevious_pathes,model_ft,update_batch_size,norm)

    model_ft.reg_params['lambda']=reg_lambda
    #model_ft = torchvision.models.alexnet()
    last_layer_index=str(len(model_ft.classifier._modules)-1)

    num_ftrs=model_ft.classifier._modules[last_layer_index].in_features 
    model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))  
   
    #check the values of omega
    sanitycheck(model_ft)
    criterion = nn.CrossEntropyLoss()
    #update the objective based params
    
    if use_gpu:
        model_ft = model_ft.cuda()
    


    #our optimizer
    optimizer_ft =Weight_Regularized_SGD(model_ft.parameters(), lr, momentum=0.9,weight_decay=weight_decay)
    #exp_dir='/esat/monkey/raljundi/pytorch/CUB11f_hebbian_finetuned'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    model_ft = train_model(model_ft, criterion, optimizer_ft,exp_lr_scheduler, lr,dset_loaders,dset_sizes,use_gpu,num_epochs,exp_dir,resume)
    
    return model_ft
def MAS_Omega_Acuumelation(dataset_path,previous_task_model_path,exp_dir,data_dir,reg_sets,reg_lambda=1,norm='L2', num_epochs=100,lr=0.0008,batch_size=200,b1=True):
    """
    In case of accumelating omega for the different tasks in the sequence, baisically to mimic the setup of standard methods where 
    the reguilizer is computed on the training set. Note that this doesn't consider our adaptation
    dataset_path=new dataset path
    exp_dir=where to save the new model
    previous_task_model_path: previous task in the sequence model path to start from 
    reg_sets: sets of examples used to compute omega. Here the default is the training set of the last task
    b1=to mimic online importance weight computation, batch size=1
    reg_lambda= regulizer hyper parameter. In object recognition it was set to 1.
    norm=the norm used to compute the gradient of the learned function
    """
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=150,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
   
    model_ft = torch.load(previous_task_model_path)
        
    if b1:
        #compute the importance with batch size of 1, to mimic the online setting
        update_batch_size=1
    else:
        update_batch_size=batch_size
    #update the omega for the previous task, accumelate it over previous omegas    
    model_ft=accumulate_MAS_weights(data_dir,reg_sets,model_ft,update_batch_size,norm)
    #set the lambda for the MAS regularizer
    model_ft.reg_params['lambda']=reg_lambda
    
    
    #get the number of features in this network and add a new task head
    last_layer_index=str(len(model_ft.classifier._modules)-1)

    num_ftrs=model_ft.classifier._modules[last_layer_index].in_features 
    model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))  
   

    
    criterion = nn.CrossEntropyLoss()
    #update the objective based params
    
    if use_gpu:
        model_ft = model_ft.cuda()
    


    #call the MAS optimizer
    optimizer_ft =Weight_Regularized_SGD(model_ft.parameters(), lr, momentum=0.9)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    #if there is a checkpoint to be resumed, in case where the training has stopped before on a given task    
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    
    #train the model
    #this training functin passes the reg params to the optimizer to be used for penalizing changes on important params
    model_ft = train_model(model_ft, criterion, optimizer_ft,exp_lr_scheduler, lr,dset_loaders,dset_sizes,use_gpu,num_epochs,exp_dir,resume)
    
    return model_ft
def update_weights_params(data_dir,reg_sets,model_ft,batch_size,norm='L2'):
    """update the importance weights based on the samples included in the reg_set. Assume starting from zero omega
    
       model_ft: the model trained on the previous task 
    """
    data_transform =  transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    

    #prepare the dataset
    dset_loaders=[]
    for data_path in reg_sets:
    
        # if so then the reg_sets is a dataset by its own, this is the case for the mnist dataset
        if data_dir is not None:
            dset = ImageFolderTrainVal(data_dir, data_path, data_transform)
        else:
            dset=torch.load(data_path)
            dset=dset['train']
        dset_loader= torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                               shuffle=False, num_workers=4)
        dset_loaders.append(dset_loader)

    
    use_gpu = torch.cuda.is_available()

    #inialize the importance params,omega, to zero
    reg_params=initialize_reg_params(model_ft)
    model_ft.reg_params=reg_params
    #define the importance weight optimizer. Actually it is only one step. It can be integrated at the end of the first task training
    optimizer_ft = MAS_Omega_update(model_ft.parameters(), lr=0.0001, momentum=0.9)
    
    if norm=='L2':
        print('********************MAS with L2 norm***************')
        #compute the imporance params
        model_ft = compute_importance_l2(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
    else:
        if norm=='vector':
            optimizer_ft=MAS_Omega_Vector_Grad_update(model_ft.parameters(), lr=0.0001, momentum=0.9)

            model_ft = compute_importance_gradient_vector(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)

        else:
            model_ft = compute_importance(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)

   
    return model_ft#
def update_sequence_MAS_weights(data_dirs,reg_sets,previous_models,model_ft,batch_size,norm='L2'):
    """updates a task in a sequence while computing omega from scratch each time on the previous tasks
       previous_models: to use their heads for compute the importance params
    """
    data_transform =  transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t=0    
    last_layer_index=str(len(model_ft.classifier._modules)-1)
    for model_path in previous_models:
        pre_model=torch.load(model_path)
        #get previous task head
        model_ft.classifier._modules[last_layer_index] = pre_model.classifier._modules[last_layer_index]

        # if data_dirs is None then the reg_sets is a dataset by its own, this is the case for the mnist dataset
        if data_dirs is not None:
            dset = ImageFolderTrainVal(data_dirs[t], reg_sets[t], data_transform)
        else:
            dset=torch.load(reg_sets[t])
            dset=dset['train']
        dset_loader= torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                               shuffle=False, num_workers=4)
       
    #=============================================================================

        use_gpu = torch.cuda.is_available()

       
        if t==0:
            #initialize to zero
            reg_params=initialize_reg_params(model_ft)
        else:
            #store previous task param
            reg_params=initialize_store_reg_params(model_ft)
        model_ft.reg_params=reg_params
        #define the importance weight optimizer. Actually it is only one step. It can be integrated at the end of the first task training
        optimizer_ft = MAS_Omega_update(model_ft.parameters(), lr=0.0001, momentum=0.9)
        #legacy code
        dset_loaders=[dset_loader]
        #compute param importance
        if norm=='L2':
            
            print('********************objective with L2 norm***************')
            model_ft = compute_importance_l2(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
        else:
            model_ft = compute_importance(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
        if t>0:
            reg_params=accumelate_reg_params(model_ft)
        
        model_ft.reg_params=reg_params
        t=t+1
    sanitycheck(model_ft)   
    return model_ft
def accumulate_MAS_weights(data_dir,reg_sets,model_ft,batch_size,norm='L2'):
    """accumelate the importance params: stores the previously computed omega, compute omega on the last previous task
            and accumelate omega resulting on  importance params for all the previous tasks
       reg_sets:either a list of files containing the samples used for computing the importance param like train or train and test
                or pytorch dataset, then train set is used
       data_dir:
    """
    data_transform =  transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    

    #prepare the dataset
    dset_loaders=[]
    for data_path in reg_sets:
    
        # if data_dir is not None then the reg_sets is a dataset by its own, this is the case for the mnist dataset
        if data_dir is not None:
            dset = ImageFolderTrainVal(data_dir, data_path, data_transform)
        else:
            dset=torch.load(data_path)
            dset=dset['train']
        dset_loader= torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                               shuffle=False, num_workers=4)
        dset_loaders.append(dset_loader)
    #=============================================================================
    
    use_gpu = torch.cuda.is_available()

    #store the previous omega, set values to zero
    reg_params=initialize_store_reg_params(model_ft)
    model_ft.reg_params=reg_params
    #define the importance weight optimizer. Actually it is only one step. It can be integrated at the end of the first task training
    optimizer_ft = MAS_Omega_update(model_ft.parameters(), lr=0.0001, momentum=0.9)
   
    if norm=='L2':
        print('********************objective with L2 norm***************')
        model_ft =compute_importance_l2(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
    else:
        model_ft =compute_importance(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
    #accumelate the new importance params  with the prviously stored ones (previous omega)
    reg_params=accumelate_reg_params(model_ft)
    model_ft.reg_params=reg_params
    sanitycheck(model_ft)   
    return model_ft

def sanitycheck(model):
    for name, param in model.named_parameters():
           
            print (name)
            if param in model.reg_params:
            
                reg_param=model.reg_params.get(param)
                omega=reg_param.get('omega')
                
                print('omega max is',omega.max())
                print('omega min is',omega.min())
                print('omega mean is',omega.mean())