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
import math
import shutil
from torch.utils.data import DataLoader
#end of imports
#
class Weight_Regularized_SGD(optim.SGD):
    r"""Implements SGD training with importance params regulization. IT inherents stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        
        super(Weight_Regularized_SGD, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)

    def __setstate__(self, state):
        super(Weight_Regularized_SGD, self).__setstate__(state)
       
        
    def step(self, reg_params,closure=None):
        """Performs a single optimization step.
        Arguments:
            reg_params: omega of all the params
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
       

        loss = None
        if closure is not None:
            loss = closure()
        
        reg_lambda=reg_params.get('lambda')
       
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
           
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
               
                #MAS PART CODE GOES HERE
                #if this param has an omega to use for regulization
                if p in reg_params:
                    
                    reg_param=reg_params.get(p)
                    #get omega for this parameter
                    omega=reg_param.get('omega')
                    #initial value when the training start
                    init_val=reg_param.get('init_val')
                    
                    curr_wegiht_val=p.data
                    #move the tensors to cuda
                    init_val=init_val.cuda()
                    omega=omega.cuda()
                    
                    #get the difference
                    weight_dif=curr_wegiht_val.add(-1,init_val)
                    #compute the MAS penalty
                    regulizer=weight_dif.mul(2*reg_lambda*omega)
                    del weight_dif
                    del curr_wegiht_val
                    del omega
                    del init_val
                    #add the MAS regulizer to the gradient
                    d_p.add_(regulizer)
                    del regulizer
                #MAS PARAT CODE ENDS
                if weight_decay != 0:
                   
                    d_p.add_(weight_decay,p.data.sign())
                   
 
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
               
                
                p.data.add_(-group['lr'], d_p)
                
        return loss#ELASTIC SGD
#from  torch.optim import Optimizer, required
#from  torch.optim import Optimizer, required
class MAS_Omega_update(optim.SGD):
    """
    Update the paramerter importance using the gradient of the function output norm. To be used at deployment time.
    reg_params:parameters omega to be updated
    batch_index,batch_size:used to keep a running average over the seen samples
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        
        super(MAS_Omega_update, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)
        
    def __setstate__(self, state):
        super(MAS_Omega_update, self).__setstate__(state)
       

    def step(self, reg_params,batch_index,batch_size,closure=None):
        """
        Performs a single parameters importance update setp
        """

        #print('************************DOING A STEP************************')
 
        loss = None
        if closure is not None:
            loss = closure()
             
        for group in self.param_groups:
   
            #if the parameter has an omega to be updated
            for p in group['params']:
          
                #print('************************ONE PARAM************************')
                
                if p.grad is None:
                    continue
               
                if p in reg_params:
                    d_p = p.grad.data
                  
                    
                    #HERE MAS IMPOERANCE UPDATE GOES
                    #get the gradient
                    unreg_dp = p.grad.data.clone()
                    reg_param=reg_params.get(p)
                    
                    zero=torch.FloatTensor(p.data.size()).zero_()
                    #get parameter omega
                    omega=reg_param.get('omega')
                    omega=omega.cuda()
    
                    
                    #sum up the magnitude of the gradient
                    prev_size=batch_index*batch_size
                    curr_size=(batch_index+1)*batch_size
                    omega=omega.mul(prev_size)
                    
                    omega=omega.add(unreg_dp.abs_())
                    #update omega value
                    omega=omega.div(curr_size)
                    if omega.equal(zero.cuda()):
                        print('omega after zero')

                    reg_param['omega']=omega
                   
                    reg_params[p]=reg_param
                    #HERE MAS IMPOERANCE UPDATE ENDS
        return loss#HAS NOTHING TO DO

  
class MAS_Omega_Vector_Grad_update(optim.SGD):
    """
    Update the paramerter importance using the gradient of the function output. To be used at deployment time.
    reg_params:parameters omega to be updated
    batch_index,batch_size:used to keep a running average over the seen samples
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        
        super(MAS_Omega_Vector_Grad_update, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)
        
    def __setstate__(self, state):
        super(MAS_Omega_Vector_Grad_update, self).__setstate__(state)
       

    def step(self, reg_params,intermediate,batch_index,batch_size,closure=None):
        """
        Performs a single parameters importance update setp
        """

        #print('************************DOING A STEP************************')

        loss = None
        if closure is not None:
            loss = closure()
        index=0
     
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
    
            for p in group['params']:
          
                #print('************************ONE PARAM************************')
                
                if p.grad is None:
                    continue
                
                if p in reg_params:
                    d_p = p.grad.data
                    unreg_dp = p.grad.data.clone()
                    #HERE MAS CODE GOES
                    reg_param=reg_params.get(p)
                    
                    zero=torch.FloatTensor(p.data.size()).zero_()
                    omega=reg_param.get('omega')
                    omega=omega.cuda()
    
                    
                    #get the magnitude of the gradient
                    if intermediate:
                        if 'w' in reg_param.keys():
                            w=reg_param.get('w')
                        else:
                            w=torch.FloatTensor(p.data.size()).zero_()
                        w=w.cuda()
                        w=w.add(unreg_dp.abs_())
                        reg_param['w']=w
                    else:
                       
                       #sum the magnitude of the gradients
                        w=reg_param.get('w')
                        prev_size=batch_index*batch_size
                        curr_size=(batch_index+1)*batch_size
                        omega=omega.mul(prev_size)
                        omega=omega.add(w)
                        omega=omega.div(curr_size)
                        reg_param['w']=zero.cuda()
                        
                        if omega.equal(zero.cuda()):
                            print('omega after zero')

                    reg_param['omega']=omega
                    #pdb.set_trace()
                    reg_params[p]=reg_param
                index+=1
        return loss
#importance_dictionary: contains all the information needed for computing the w and omega
  

  
def train_model(model, criterion, optimizer, lr_scheduler,lr,dset_loaders,dset_sizes,use_gpu, num_epochs,exp_dir='./',resume=''):
    """Train a given model using MAS optimizer. The only unique thing is that it passes the importnace params to the optimizer"""
    print('dictoinary length'+str(len(dset_loaders)))
    #reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])

        
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
                #FOR MNIST DATASET
                inputs=inputs.squeeze()
                
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    #pass omega to the optimizer to use for penalizing the weights changes
                    optimizer.step(model.reg_params)

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
                
      
        epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
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
#importance_dictionary: contains all the information needed for computing the w and omega


def compute_importance_l2(model, optimizer, lr_scheduler,dset_loaders,use_gpu):
    """Mimic the depoloyment setup where the model is applied on some samples and those are used to update the importance params
       Uses the L2norm of the function output. This is what we MAS uses as default
    """
    print('dictoinary length'+str(len(dset_loaders)))
    #reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0
    
    
    
        
    epoch=1
    #it does nothing here
    optimizer = lr_scheduler(optimizer, epoch,1)
    model.eval()  # Set model to training mode so we get the gradient


    running_loss = 0.0
    running_corrects = 0
   
    # Iterate over data.
    index=0
    for dset_loader in dset_loaders:
        for data in dset_loader:
            # get the inputs
            inputs, labels = data
            if inputs.size(1)==1 and len(inputs.size())==3:
                
                #for mnist, there is no channel 
                #and  to avoid problems we remove that additional dimension generated by pytorch transformation
                inputs=inputs.view(inputs.size(0),inputs.size(2))            
            # wrap them in Variable
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)


        
            #compute the L2 norm of output 
            Target_zeros=torch.zeros(outputs.size())
            Target_zeros=Target_zeros.cuda()
            Target_zeros=Variable(Target_zeros)
            #note no avereging is happening here
            loss = torch.nn.MSELoss(size_average=False)

            targets = loss(outputs,Target_zeros) 
            #compute the gradients
            targets.backward()

            #update the parameters importance
            optimizer.step(model.reg_params,index,labels.size(0))
            print('batch number ',index)
            #nessecary index to keep the running average
            index+=1
   
    return model


def compute_importance(model, optimizer, lr_scheduler,dset_loaders,use_gpu):
    """Mimic the depoloyment setup where the model is applied on some samples and those are used to update the importance params
       Uses the L1norm of the function output
    """
    print('dictoinary length'+str(len(dset_loaders)))
   
    since = time.time()

    best_model = model
    best_acc = 0.0
    
    #pdb.set_trace()
    

        
    epoch=1
    #it does nothing here, can be removed
    optimizer = lr_scheduler(optimizer, epoch,1)
    model.eval()  # Set model to training mode so we get the gradient


    running_loss = 0.0
    running_corrects = 0
   
    # Iterate over data.
    index=0
    for dset_loader in dset_loaders:
        #pdb.set_trace()
        for data in dset_loader:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameters gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
      

           #compute the L1 norm of the function output
        
            Target_zeros=torch.zeros(outputs.size())
            Target_zeros=Target_zeros.cuda()
            Target_zeros=Variable(Target_zeros,requires_grad=False)
       
            loss = torch.nn.L1Loss(size_average=False)

            targets = loss(outputs,Target_zeros) 
            #compute gradients
            targets.backward()
        
            
            print('batch number ',index)
            #update parameters importance
            optimizer.step(model.reg_params,index,labels.size(0))
            #nessecary index to keep the running average
            index+=1
   
    return model


def compute_importance_gradient_vector(model, optimizer, lr_scheduler,dset_loaders,use_gpu):
    """Mimic the depoloyment setup where the model is applied on some samples and those are used to update the importance params
       Uses the gradient of the function output
    """
    print('dictoinary length'+str(len(dset_loaders)))
    #reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0
    
    
    
        
    epoch=1
    optimizer = lr_scheduler(optimizer, epoch,1)
    model.eval()  # Set model to training mode so we get the gradient


    running_loss = 0.0
    running_corrects = 0
   
    # Iterate over data.
    index=0
    for dset_loader in dset_loaders:
        for data in dset_loader:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
     
           
            for output_i in range(0,outputs.size(1)):
                Target_zeros=torch.zeros(outputs.size())
                Target_zeros=Target_zeros.cuda()
                Target_zeros[:,output_i]=1
                Target_zeros=Variable(Target_zeros,requires_grad=False)
                targets=torch.sum(outputs*Target_zeros)
                if output_i==(outputs.size(1)-1):
                    targets.backward()
                else:
                    targets.backward(retain_graph=True )
                    
                optimizer.step(model.reg_params,True,index,labels.size(0))
                optimizer.zero_grad()
            
        #print('step')
            optimizer.step(model.reg_params,False,index,labels.size(0))
            print('batch number ',index)
            index+=1
   
    return model
def initialize_reg_params(model,freeze_layers=[]):
    """initialize an omega for each parameter to zero"""
    reg_params={}
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            print('initializing param',name)
            omega=torch.FloatTensor(param.size()).zero_()
            init_val=param.data.clone()
            reg_param={}
            reg_param['omega'] = omega
            #initialize the initial value to that before starting training
            reg_param['init_val'] = init_val
            reg_params[param]=reg_param
    return reg_params
   


def initialize_store_reg_params(model,freeze_layers=[]):
    """set omega to zero but after storing its value in a temp omega in which later we can accumolate them both"""
    reg_params=model.reg_params
    for name, param in model.named_parameters():
        #in case there some layers that are not trained
        if not name in freeze_layers:
            if param in reg_params:
                reg_param=reg_params.get(param)
                print('storing previous omega',name)
                prev_omega=reg_param.get('omega')
                new_omega=torch.FloatTensor(param.size()).zero_()
                init_val=param.data.clone()
                reg_param['prev_omega']=prev_omega   
                reg_param['omega'] = new_omega
                
                #initialize the initial value to that before starting training
                reg_param['init_val'] = init_val
                reg_params[param]=reg_param
                
        else:
            if param in reg_params: 
                reg_param=reg_params.get(param)
                print('removing unused omega',name)
                del reg_param['omega'] 
                del reg_params[param]
    return reg_params
   


def accumelate_reg_params(model,freeze_layers=[]):
    """accumelate the newly computed omega with the previously stroed one from the old previous tasks"""
    reg_params=model.reg_params
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in reg_params:
                reg_param=reg_params.get(param)
                print('restoring previous omega',name)
                prev_omega=reg_param.get('prev_omega')
                prev_omega=prev_omega.cuda()
                
                new_omega=(reg_param.get('omega')).cuda()
                acc_omega=torch.add(prev_omega,new_omega)
                
                del reg_param['prev_omega']
                reg_param['omega'] = acc_omega
               
                reg_params[param]=reg_param
                del acc_omega
                del new_omega
                del prev_omega
        else:
            if param in reg_params: 
                reg_param=reg_params.get(param)
                print('removing unused omega',name)
                del reg_param['omega'] 
                del reg_params[param]             
    return reg_params
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    #best_model = copy.deepcopy(model)
    torch.save(state, filename)
   