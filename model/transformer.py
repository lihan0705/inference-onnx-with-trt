import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.optim as optim
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(320, 20, kernel_size=5)

        self.conv3 = nn.Conv2d(10, 40, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(80, 160, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(160, 320, kernel_size=3, padding=1)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, input_dict):
        x = input_dict['x']
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)


        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)