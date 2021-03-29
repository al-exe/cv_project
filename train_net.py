from __future__ import print_function, division

import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from model import Net

def main():
    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    # Create the training data generator
    batch_size = 100
    im_height = 64
    im_width = 64
    num_epochs = 15

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255))))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255))))
        ]),
    }

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(data_dir / 'train' , data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(data_dir / 'val/labeled_val' , data_transforms['val'])

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=64, pin_memory=True)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc)
                else:
                    val_losses.append(epoch_loss)
                    val_accs.append(epoch_acc)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # Plot losses
        plt.figure()
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.title('Loss')
        plt.legend(frameon=False)
        plt.savefig('loss.png')
        # Plot accuracies
        plt.figure()
        plt.plot(train_accs, label='Training accuracy')
        plt.plot(val_accs, label='Validation accuracy')
        plt.xlabel('Epoch')
        plt.title('Accuracy')  
        plt.legend(frameon=False)
        plt.savefig('accuracy.png')
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    model_ft = models.resnet50(pretrained=True)
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 200)
    # num_ftrs = model_ft.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model_ft.fc = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)
    # model_ft = torch.nn.DataParallel(model_ft, device_ids=[0])

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # scheduler = lr_scheduler.OneCycleLR(optimizer_ft, max_lr=0.02, steps_per_epoch=len(dataloaders['train']),
    #                     epochs=num_epochs, div_factor=10, final_div_factor=10,
    #                     pct_start=10/num_epochs)

    model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler,
                       num_epochs=num_epochs)
    torch.save({
    'net': model_ft.state_dict(),
    }, 'latest.pt')


if __name__ == '__main__':
    main()
