import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import struct
import time

n_in = 9216
n_out = 128
batch_size = 256




def test1(n_in, n_out, batch_size, device='cuda'):
    device = torch.device(device)

    fc = nn.Linear(n_in, n_out, bias=False).to(device=device)
    for parameter in fc.parameters():
        parameter.requires_grad = False  # This increases the speed of result 4 like CRAZY

    input_dummy = torch.zeros(n_in).to(device=device)
    result_dummy = fc(input_dummy)  # For some reason (maybe cudnn) the first iteration is much slower (1000x)

    inputs = torch.empty(batch_size, n_in).normal_().to(device=device)

    n_iter = 100
    start = time.clock()
    for i in range(n_iter):
        result = fc.forward(inputs)

    elapsed_time = time.clock() - start
    print("Elapsed time: " + "{0:0.3f}".format(elapsed_time) + "s")




def test2(n_in, n_out, batch_size, device='cuda'):
    device = torch.device(device)

    fc = nn.Linear(n_in, n_out, bias=False).to(device=device)
    for parameter in fc.parameters():
        parameter.requires_grad = False  # This increases the speed of result 4 like CRAZY

    inputs_dummy = torch.zeros(batch_size, n_in).to(device=device)
    inputs_dummy_expanded = inputs_dummy.unsqueeze(1).expand(batch_size, n_out, n_in)
    W = fc.weight.data
    result_dummy = (W * inputs_dummy_expanded).sum(dim=2)  # For some reason (maybe cudnn) the first iteration is much slower (1000x)

    inputs = torch.empty(batch_size, n_in).normal_().to(device=device)
    torch.cuda.synchronize()
    n_iter = 100
    start = time.clock()
    for i in range(n_iter):
        inputs_expanded = inputs.unsqueeze(1).expand(batch_size, n_out, n_in)
        result = (W * inputs_expanded).sum(dim=2)
        torch.cuda.synchronize()
    elapsed_time = time.clock() - start
    print(repr(i) + " - Elapsed time: " + "{0:0.3f}".format(elapsed_time) + "s")


def main():
    n_in = 9216
    n_out = 128
    batch_size = 256
    test1(n_in, n_out, batch_size, 'cuda')
    test2(n_in, n_out, batch_size, 'cuda')
    return


if __name__ == '__main__':
    main()

