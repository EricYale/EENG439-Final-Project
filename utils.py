import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
import argparse



# training model in pytorch
def train(model, device, train_loader, optimizer, criterion, epoch, train_losses, train_accuracies):
    model.train()
    train_loss = 0
    correct = 0
    total_train = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)

        predicted = (output > 0.5).float()
        total_train += target.size(0)
        correct += (predicted == target).sum().item()

    #calculating the total loss
    train_loss = ((train_loss)/len(train_loader.dataset))
    train_losses.append(train_loss)

    #accuracy
    accuracy = (100*correct)/len(train_loader.dataset)
    train_accuracies.append(accuracy)

    #logging the result
    print("Train Epoch: %d Train Loss: %.4f. Train Accuracy: %.2f." % (epoch, train_loss, accuracy))



# testing model in pytorch
def test(model, device, test_loader, criterion, epoch, test_losses, test_accuracies):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    #test loss calculation
    test_loss += criterion(output, target).item() * data.size(0)

    test_losses.append(test_loss)

    #calculating the accuracy in the validation step
    accuracy = (100*correct)/len(test_loader.dataset)
    test_accuracies.append(accuracy)

    #logging the results
    print("Test Epoch: %d Test Loss: %.4f Test Accuracy: %.2f." %
          (epoch, test_loss, accuracy))


# model fitting in pytorch
def fit(model, device, train_loader, test_loader, optimizer, criterion, no_of_epochs):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(0, no_of_epochs):
        print(f"epoch: {epoch}, pct : {np.round(epoch/no_of_epochs,2)}")
        train(model, device, train_loader, optimizer,
              criterion, epoch, train_losses, train_accuracies)
        test(model, device, test_loader, criterion,
             epoch, test_losses, test_accuracies)
    return train_losses, test_losses, train_accuracies, test_accuracies
