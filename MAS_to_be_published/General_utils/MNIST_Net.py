
import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
class MNIST_Net(nn.Module):
    def __init__(self,num_classes=10,hidden_size=256):
        super(MNIST_Net, self).__init__()

        self.classifier = nn.Sequential(
           
            nn.Linear( 28 * 28, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        x = self.classifier(x)
        return x
import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
class MNIST_Net_buttelNeck(nn.Module):
    def __init__(self,num_classes=10,hidden_size=256,buttel_neck=10):
        super(MNIST_Net_buttelNeck, self).__init__()

        self.classifier = nn.Sequential(
           
            nn.Linear( 28 * 28, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, buttel_neck),
            nn.ReLU(inplace=True),
            nn.Linear(buttel_neck, num_classes))

    def forward(self, x):
        x = self.classifier(x)
        return x