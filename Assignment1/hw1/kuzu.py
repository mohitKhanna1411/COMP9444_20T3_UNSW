# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.linear_layer = nn.Linear(28*28, 10)
        # INSERT CODE HERE

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear_layer(x)
        x = F.log_softmax(x, dim=1)
        return x  # CHANGE CODE HERE


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.fc_1 = nn.Linear(28*28, 16*16)
        self.fc_2 = nn.Linear(16*16, 20)
        # INSERT CODE HERE

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc_1(x)
        x = torch.tanh(x)
        x = self.fc_2(x)
        x = torch.log_softmax(x, dim=1)
        return x  # CHANGE CODE HERE


class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Conv2d(
            in_channels=64, out_channels=24, kernel_size=5, padding=2)
        self.fc_1 = nn.Linear(1176, 159)
        self.fc_2 = nn.Linear(159, 10)
        # INSERT CODE HERE

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)  # flattening the inputs.
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.log_softmax(x, dim=1)
        return x  # CHANGE CODE HERE
