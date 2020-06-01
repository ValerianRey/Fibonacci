import torch
import torch.nn as nn


class Rounding(nn.Module):
    def __init__(self):
        super(Rounding, self).__init__()

    def forward(self, x):
        return torch.round(x)

