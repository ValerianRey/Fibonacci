from __future__ import print_function
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, non_linearity=nn.ReLU):
        super(Net, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            non_linearity(),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            non_linearity(),
            nn.Dropout2d(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.seq(x)


class NetNoPool(nn.Module):
    def __init__(self, non_linearity=nn.ReLU):
        super(NetNoPool, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            non_linearity(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            non_linearity(),
            nn.Dropout2d(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.seq(x)