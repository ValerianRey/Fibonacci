import torch.nn as nn


supported_modules = {nn.Linear, nn.Conv2d}
batch_norm_modules = {nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d}
