from utils import * 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg11

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
print('on: ', device)


transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Mean and std for CIFAR-10
])

"Model Parameters"
model = vgg11(pretrained = True).to(device)
#modify final layer 
model.classifier[6] = nn.Sequential(
    nn.Linear(model.classifier[6].in_features, 1),
    nn.Sigmoid()
)
model = model.to(device)
criterion = nn.BCELoss() #Binary Cross Entropy 
optimizer = optim.SGD(model.parameters(),lr = 0.01, momentum = 0.9, weight_decay = 5e-4)

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

"----------------------Training"
#def fit(model, device, train_loader, test_loader, optimizer, criterion, no_of_epochs):
train_losses,test_losses,train_accuracies, test_accuracies = fit(model,device,trainloader,testloader,optimizer,criterion,150)


"-------------------------------------------------------Visualization-------------------------------"
import matplotlib.pyplot as plt

# Assuming you have these values stored in lists
epochs = range(1, len(VGGtrain_losses) + 1)

import matplotlib.pyplot as plt

# Move losses to CPU and convert to NumPy if they are tensors
VGGtrain_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in VGGtrain_losses]
VGGtest_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in VGGtest_losses]

# Plotting
epochs = range(1, len(VGGtrain_losses) + 1)

plt.figure(figsize=(12, 6))

# Subplot 1: Training and Test Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, VGGtrain_losses, label='Training Loss')
plt.plot(epochs, VGGtest_losses, label='Test Loss')
plt.title('Training and Test Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Display the plot
plt.show()

# Plot training and test accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, VGGtrain_accuracies, label='Training Accuracy')
plt.plot(epochs, VGGtest_accuracies, label='Test Accuracy')
plt.title('Training and Test Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
