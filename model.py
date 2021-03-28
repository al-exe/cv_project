import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self, num_classes, im_height, im_width):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(im_height * im_width * 3, 128)
        self.layer2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class APNet(nn.Module): 
    def __init__(self, num_classes, im_height, im_width):
        super(APNet, self).__init__()
        self.c1 = nn.Conv2d(3, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.c3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.d1 = nn.Dropout(p=0.3, inplace=False)
        self.d2 = nn.Dropout(p=0.1, inplace=False)
        self.d3 = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x):
        x = self.pool(F.elu(self.c1(x)))
        x = self.d3(x)
        x = self.pool(F.elu(self.c2(x)))
        x = self.d1(x)
        x = self.pool(F.elu(self.c3(x)))
        x = self.d2(x)
        x = x.view(-1, 128 * 2 * 2)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x
