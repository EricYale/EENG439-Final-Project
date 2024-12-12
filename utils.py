
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras 
from utils import * 
import logging

def download_data(batch_size, img_size): 
    """_summary_

    Args:
        batch_size (int): number of batches
        img_size (tuple): for a square image

    Returns:
        _type_: train_dataset, validation_dataset, test_dataset
    """
    BATCH_SIZE = batch_size
    IMG_SIZE = img_size
    train_dir = 'data/output/train'
    validation_dir = 'data/output/val'
    test_dir = 'data/output/test'

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                    shuffle=True,
                                                                    batch_size=BATCH_SIZE,
                                                                    image_size=IMG_SIZE)

    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                    shuffle=True,
                                                                    batch_size=BATCH_SIZE,
                                                                    image_size=IMG_SIZE)
    return train_dataset, validation_dataset, test_dataset

def generate_sample_images(): 
    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        
def configure_data(train_dataset, validation_dataset, test_dataset): 
    """
    use buffered prefetching to load images from disk without having I/O become blocking 
    """
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, validation_dataset, test_dataset

    
def plot_acc_loss(acc, val_acc, loss, val_loss, initial_epochs = 10): 
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    
    
def pred_images(test_dataset, model, class_names): 
    # Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[predictions[i]])
        plt.axis("off")

def log_epoch_results(epoch, total_epochs, steps, step_time, accuracy, loss, val_accuracy, val_loss):
    log_message = (
        f"Epoch {epoch}/{total_epochs}\n"
        f"{steps}/{steps} ━ {step_time}s {step_time / steps:.0f}ms/step - "
        f"accuracy: {accuracy:.4f} - loss: {loss:.4f} - "
        f"val_accuracy: {val_accuracy:.4f} - val_loss: {val_loss:.4f}"
    )
    logging.info(log_message)


# import sys
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# import numpy as np
# import torchvision
# import torchvision.transforms as transforms

# import os
# import argparse



# # training model in pytorch
# def train(model, device, train_loader, optimizer, criterion, epoch, train_losses, train_accuracies):
#     model.train()
#     train_loss = 0
#     correct = 0
#     total_train = 0

#     for data, target in train_loader:
#         data, target = data.to(device), target.to(device)
#         target = target.float().unsqueeze(1).to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()*data.size(0)  

#         predicted = (output > 0.5).float()
#         total_train += target.size(0)
#         correct += (predicted == target).sum().item()

#     #calculating the total loss
#     train_loss = ((train_loss)/len(train_loader.dataset))
#     train_losses.append(train_loss)

#     #accuracy
#     accuracy = (100*correct)/len(train_loader.dataset)
#     train_accuracies.append(accuracy)

#     #logging the result
#     print("Train Epoch: %d Train Loss: %.4f. Train Accuracy: %.2f." % (epoch, train_loss, accuracy))



# # testing model in pytorch
# def test(model, device, test_loader, criterion, epoch, test_losses, test_accuracies):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             target = target.float().unsqueeze(1).to(device)
#             output = model(data)
#             test_loss += criterion(output, target)
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     #test loss calculation
#     test_loss += criterion(output, target).item() * data.size(0)

#     test_losses.append(test_loss)

#     #calculating the accuracy in the validation step
#     accuracy = (100*correct)/len(test_loader.dataset)
#     test_accuracies.append(accuracy)

#     #logging the results
#     print("Test Epoch: %d Test Loss: %.4f Test Accuracy: %.2f." %
#           (epoch, test_loss, accuracy))


# # model fitting in pytorch
# def fit(model, device, train_loader, test_loader, optimizer, criterion, no_of_epochs):
#     train_losses = []
#     test_losses = []
#     train_accuracies = []
#     test_accuracies = []
#     for epoch in range(0, no_of_epochs):
#         print(f"epoch: {epoch}, pct : {np.round(epoch/no_of_epochs,2)}")
#         train(model, device, train_loader, optimizer,
#               criterion, epoch, train_losses, train_accuracies)
#         test(model, device, test_loader, criterion,
#              epoch, test_losses, test_accuracies)
#     return train_losses, test_losses, train_accuracies, test_accuracies
