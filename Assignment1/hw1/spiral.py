# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.fc_1 = nn.Linear(2, num_hid)
        self.fc_2 = nn.Linear(num_hid, 1)
        # INSERT CODE HERE

    def forward(self, input):
        x = input[:, 0]
        y = input[:, 1]
        # convertion to polar co-ordinates from the inputs.
        sqaure_sum = (x**2) + (y**2)
        z = torch.sqrt(sqaure_sum)
        r = z.reshape(-1, 1)  # final r

        a = torch.atan2(y, x)
        a = a.reshape(-1, 1)  # final a
        temp = torch.cat((r, a), 1)

        hidden_1 = self.fc_1(temp)
        self.layer_1_sum = torch.tanh(hidden_1)
        hidden_2 = self.fc_2(self.layer_1_sum)
        output = torch.sigmoid(hidden_2)
        return output


class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.fc_1 = nn.Linear(2, num_hid)
        self.fc_2 = nn.Linear(num_hid, num_hid)
        self.fc_3 = nn.Linear(num_hid, 1)
        # INSERT CODE HERE

    def forward(self, input):
        hid_1 = self.fc_1(input)
        self.layer_1_sum = torch.tanh(hid_1)
        hid_2 = self.fc_2(self.layer_1_sum)
        self.layer_2_sum = torch.tanh(hid_2)
        hid_3 = self.fc_3(self.layer_2_sum)
        output = torch.sigmoid(hid_3)
        return output


def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)

    with torch.no_grad():
        net.eval()
        output = net(grid)
        if layer == 1:
            pred = (net.layer_1_sum[:, node] >= 0).float()
        elif layer == 2:
            pred = (net.layer_2_sum[:, node] >= 0).float()

        plt.clf()
        plt.pcolormesh(xrange, yrange, pred.cpu().view(
            yrange.size()[0], xrange.size()[0]), cmap='Wistia')
    # plt.clf()
    # INSERT CODE HERE
