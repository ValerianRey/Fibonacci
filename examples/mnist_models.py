from __future__ import print_function
import torch.nn as nn
from examples.supported_modules import supported_modules
from examples.print_util import Color, print_layer

n_iter = 0


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.seq(x)

    def print(self, how='long'):
        if how == 'no':
            return
        elif how == 'long':
            print_data = True
        else:
            print_data = False

        layers_to_print = []
        names_to_print = []

        for layer in self.seq:
            if type(layer) in supported_modules:
                layers_to_print.append(layer)
                names_to_print.append(layer.__class__.__name__)

        for name, layer in zip(names_to_print, layers_to_print):
            print_layer(name, layer, print_data=print_data)

