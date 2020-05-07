from __future__ import print_function
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, dropout=False):
        super(LeNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25) if dropout else nn.Identity(),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Dropout2d(0.5) if dropout else nn.Identity(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.seq(x)
