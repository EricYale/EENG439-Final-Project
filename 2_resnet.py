from utils import * 
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from torchvision.models import vgg11, resnet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
print('on: ', device)


netRES= resnet18(num_classes = 10).to(device)
criterion = nn.BCELoss()
#optimizerRES = optim.SGD(netRES.parameters(),lr = 0.01, momentum = 0.9, weight_decay = 5e-4)
optimizerRES =  optim.Adam(netRES.parameters(),lr=0.0001)