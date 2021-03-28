"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

import pathlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import APNet, Net

from torch import nn
import wandb


def main():
    # Create a pytorch dataset
    use_wandb = False
    if use_wandb:
        wandb.init(project="182-vision")

    device='cuda'

    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    # Create the training data generator
    batch_size = 32
    im_height = 64
    im_width = 64
    num_epochs = 1

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])
    train_set = torchvision.datasets.ImageFolder(data_dir / 'train', data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    val_set = torchvision.datasets.ImageFolder(data_dir / 'val', data_transforms)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                        shuffle=True, num_workers=4, pin_memory=True)

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=2048, out_features=200, bias=True)
    print(model)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    # for i in range(num_epochs):
    #     train_total, train_correct = 0,0
    #     for idx, (inputs, targets) in enumerate(train_loader):
    #         optim.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optim.step()
    #         _, predicted = outputs.max(1)
    #         train_total += targets.size(0)
    #         train_correct += predicted.eq(targets).sum().item()
    #         print("\r", end='')
    #         print(f'training {100 * idx / len(train_loader):.2f}%: {train_correct / train_total:.3f}', end='')
    #     torch.save({
    #         'net': model.state_dict(),
    #     }, 'latest.pt')
    epochs = 1
    steps = 0
    train_loss = 0
    print_every = 10
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if use_wandb:
                wandb.log({'epoch': epoch, 'training loss': train_loss})

            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        val_loss += batch_loss.item()

                        if use_wandb:
                            wandb.log({'epoch': epoch, 'val loss': val_loss})

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(train_loss/len(train_loader))
                val_losses.append(val_loss/len(val_loader))                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {train_loss/print_every:.3f}.. "
                    f"Validation loss: {val_loss/len(val_loader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(val_loader):.3f}")
                train_loss = 0
                model.train()
        torch.save({
            'net': model.state_dict(),
        }, 'latest.pt')

if __name__ == '__main__':
    main()

