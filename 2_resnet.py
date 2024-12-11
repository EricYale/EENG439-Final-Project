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
from torchvision.models import resnet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
print('on: ', device)


model= resnet18(num_classes = 10).to(device)
criterion = nn.BCELoss()
#optimizerRES = optim.SGD(netRES.parameters(),lr = 0.01, momentum = 0.9, weight_decay = 5e-4)
optimizerRES =  optim.Adam(model.parameters(),lr=0.0001)


"DataLoader"
OUTPUT_DIR = "data/output/"
train_data = datasets.ImageFolder(
    OUTPUT_DIR + "train", transform=transformation
)
val_data = datasets.ImageFolder(
    OUTPUT_DIR + "val", transform=transformation
)
test_data = datasets.ImageFolder(
    OUTPUT_DIR + "test", transform=transformation
)

BATCH_SIZE = 128
trainloader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)
valloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False
)

"Training"
#def fit(model, device, train_loader, test_loader, optimizer, criterion, no_of_epochs):
train_losses,test_losses,train_accuracies, test_accuracies = fit(model,device,trainloader,testloader,optimizer,criterion,150)