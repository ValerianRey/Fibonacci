from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from examples.print_util import Color, print_layer

n_iter = 0


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 input channel, 32 output channels, 3x3 convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 3)  # 32 input channel, 64 output channels, 3x3 convolution kernel
        self.dropout1 = nn.Dropout2d(0.25)  # Drops 25% of the channels randomly when training
        self.fc1 = nn.Linear(9216, 128)  # fully connected linear layer of 9216 input nodes and 128 output nodes
        self.dropout2 = nn.Dropout2d(0.5)  # Drops 50% of the channels randomly when training
        self.fc2 = nn.Linear(128, 10)  # fully connected linear layer of 128 input nodes and 10 output nodes (digit probabilities)

    def forward(self, x):
        # Input size is commented on the right
        x = self.conv1(x)  # N x 1 x 28 x 28
        x = F.relu(x)  # N x 32 x 26 x 26
        x = self.conv2(x)  # N x 64 x 24 x 24
        x = F.max_pool2d(x, 2)  # N x 64 x 12 x 12
        x = self.dropout1(x)  # ? N x 64 x 12 x 12
        x = torch.flatten(x, 1)  # ? N x 64 x 12 x 12
        x = self.fc1(x)  # N x 9216
        x = F.relu(x)  # N x 128
        x = self.dropout2(x)  # N x 128
        x = self.fc2(x)  # N x 128
        output = x  # F.log_softmax(x, dim=1)  # N x 10
        return output  # N x 10

    def print(self, how='long'):
        if how == 'no':
            return
        elif how == 'long':
            print_data = True
        else:
            print_data = False

        layers_to_print = []
        names_to_print = []

        layers_to_print.append(self.conv1)
        layers_to_print.append(self.conv2)
        layers_to_print.append(self.fc1)
        layers_to_print.append(self.fc2)
        names_to_print.append("Conv2D 1")
        names_to_print.append("Conv2D 2")
        names_to_print.append("Linear 1")
        names_to_print.append("Linear 2")

        for name, layer in zip(names_to_print, layers_to_print):
            print_layer(name, layer, print_data=print_data)

