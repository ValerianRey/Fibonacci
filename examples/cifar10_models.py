from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


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


# Pre-activation version of the BasicBlock. The residual part has been removed to keep everything sequential.
def pre_act_block(in_planes, planes, affine_batch_norm, stride=1):
    modules = [
        nn.BatchNorm2d(in_planes, affine=affine_batch_norm),
        nn.ReLU(),
        nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(planes, affine=affine_batch_norm),
        nn.ReLU(),
        nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)]
    return modules


# Pre-activation version of the original Bottleneck module. The residual part has been removed to keep everything sequential.
def pre_act_bottleneck(in_planes, planes, affine_batch_norm, stride=1):
    expansion = 4
    modules = [
        nn.BatchNorm2d(in_planes, affine=affine_batch_norm),
        nn.ReLU(),
        nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
        nn.BatchNorm2d(planes, affine=affine_batch_norm),
        nn.ReLU(),
        nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(planes, affine=affine_batch_norm),
        nn.ReLU(),
        nn.Conv2d(planes, expansion * planes, kernel_size=1, bias=False)]
    return modules


class PreActResNet(nn.Module):
    def __init__(self, block, expansion, num_blocks, affine_batch_norm, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            *self._make_layer(block, expansion, 64, num_blocks[0], affine_batch_norm, stride=1),
            *self._make_layer(block, expansion, 128, num_blocks[1], affine_batch_norm, stride=2),
            *self._make_layer(block, expansion, 256, num_blocks[2], affine_batch_norm, stride=2),
            *self._make_layer(block, expansion, 512, num_blocks[3], affine_batch_norm, stride=2),
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512 * expansion, num_classes)
        )

    def _make_layer(self, block, expansion, planes, num_blocks, affine_batch_norm, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.extend(block(self.in_planes, planes, affine_batch_norm, stride))
            self.in_planes = planes * expansion
        return layers

    def forward(self, x):
        return self.seq(x)


def parn(depth=18, affine_batch_norm=True):
    if depth == 18:
        return PreActResNet(pre_act_block, 1, [2, 2, 2, 2], affine_batch_norm)
    elif depth == 34:
        return PreActResNet(pre_act_block, 1, [3, 4, 6, 3], affine_batch_norm)
    elif depth == 50:
        return PreActResNet(pre_act_bottleneck, 4, [3, 4, 6, 3], affine_batch_norm)
    elif depth == 101:
        return PreActResNet(pre_act_bottleneck, 4, [3, 4, 23, 3], affine_batch_norm)
    elif depth == 152:
        return PreActResNet(pre_act_bottleneck, 4, [3, 8, 36, 3], affine_batch_norm)
    else:
        raise ValueError("Depth {} not supported. Use 18, 34, 50, 101 or 152.".format(depth))


