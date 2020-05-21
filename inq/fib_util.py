import torch
import numpy as np


def binary_int(num, bits=8):
    code = bin(num)[2:]
    return '0' * (bits - len(code)) + code


def int_from_bin(code):
    return int(code, 2)


# Gives the Fibonacci-valid int number that is the closest (down) to our int number 'num'
def fib_code_int_down(num, bits=8):
    code = list(binary_int(num, bits))
    count = 0
    for i in range(len(code)):
        if code[i] == '1':
            count += 1
            if count >= 2:
                code[i] = '0' # Remove the problem
                # Then we need to make the number as big as possible by placing only 10101010101... until the end of the floating point representation
                one_next = True
                for j in range(i+1, len(code)):
                    if one_next:
                        code[j] = '1'
                        one_next = False
                    else:
                        code[j] = '0'
                        one_next = True
                break
        else:
            count = 0

    code = ''.join(code)
    return int_from_bin(code)


# Gives the Fibonacci-valid int number that is the closest (up) to our int number 'num'
def fib_code_int_up(num, bits=8):
    code = list(binary_int(num, bits))
    count = 0
    for i in range(len(code)):
        if code[i] == '1':
            count += 1
            if count >= 2:

                one_next = True
                for j in range(i - 2, -1, -1):
                    if one_next:
                        code[j] = '1'
                        one_next = False
                    else:
                        code[j] = '0'
                        one_next = True
                    if j > 0 and code[j - 1] == '0':
                        break

                i -= 1
                while i < len(code):
                    code[i] = '0'  # Remove the problem (and all subsequent problems on the right)
                    i += 1
                break

        else:
            count = 0

    code = ''.join(code)
    return int_from_bin(code)


# Gives the best Fibonacci-valid int approximation for 'num'
def fib_code_int(num, bits=8):
    if num < 0:
        # print("Clamping " + repr(num) + " to 0")
        num = 0
    elif num >= 2 ** bits:
        # print("Clamping " + repr(num) + " to " + repr(2 ** bits - 1))
        num = 2 ** bits - 1
    down = fib_code_int_down(num, bits)
    up = fib_code_int_up(num, bits)
    dist_down = abs(num - down)
    dist_up = abs(num - up)
    if dist_down < dist_up:
        return down
    else:
        return up


def is_fib(num, bits=8):
    if num < 0 or num >= 2 ** bits:  # An int weight can be outside of the range before it is fib quantized (this is intended)
        return False
    last_one = False
    for bit in binary_int(num, bits):
        if bit == '1':
            if last_one:
                return False
            last_one = True
        else:
            last_one = False
    return True


# Returns a tensor of the same shape as x with a 1 at each fib number and a 0 at each non-fib number
def is_fib_tensor(x, bits=8):
    return torch.zeros_like(x).copy_(x).detach().int().cpu().apply_(lambda num: is_fib(num, bits))


def proportion_fib(x, bits=8):
    with torch.no_grad():
        return (is_fib_tensor(x, bits).sum() / np.prod(x.shape)).item()


def fib_distances(x, bits=8):
    with torch.no_grad():
        x_fib = x.int().cpu().apply_(lambda y: fib_code_int(y, bits=bits)).float().cuda()
        distances = torch.abs(x - x_fib)
        return x_fib, distances


def fib_quantize_tensor(q_tensor, proportions, step, bits=8, strategy='quantile'):
    with torch.no_grad():
        if strategy == 'quantile':
            interpolation = 'higher'  # Higher ensures that we always at least have 'proportions[step]' weights fib encoded
            q_tensor_fib, distances = fib_distances(q_tensor, bits)
            quantile = np.quantile(distances.cpu().numpy(), proportions[step], interpolation=interpolation)
            q_tensor = torch.where(distances <= quantile, q_tensor_fib, q_tensor)
            ones = torch.ones_like(q_tensor)
            zeros = torch.zeros_like(q_tensor)
            Ts = torch.where(distances <= quantile, zeros, ones)

        elif strategy == 'reverse_quantile':
            interpolation = 'lower'  # Lower ensures that we always at least have 'proportions[step]' weights fib encoded
            if step == 0:
                proportion = proportions[0]
            elif step >= len(proportions) - 1:
                proportion = 1
            else:
                proportion = proportions[step] - proportions[step - 1]

            q_tensor_fib, distances = fib_distances(q_tensor, bits)
            quantile = np.quantile(distances.cpu().numpy(), 1 - proportion, interpolation=interpolation)
            q_tensor = torch.where(distances >= quantile, q_tensor_fib, q_tensor)
            ones = torch.ones_like(q_tensor)
            zeros = torch.zeros_like(q_tensor)
            Ts = torch.where(distances >= quantile, zeros, ones)

        elif strategy == 'random':
            if step == 0:
                proportion = proportions[0]
            elif step >= len(proportions) - 1:
                proportion = 1
            else:
                proportion = (proportions[step] - proportions[step - 1]) / (1 - proportions[step - 1])
            q_tensor_fib = q_tensor.int().cpu().apply_(lambda y: fib_code_int(y, bits=bits)).float().cuda()
            rand = torch.rand_like(q_tensor)
            ones = torch.ones_like(q_tensor)
            zeros = torch.zeros_like(q_tensor)
            Ts = torch.where(rand <= proportion, zeros, ones)
            q_tensor = torch.where(rand <= proportion, q_tensor_fib, q_tensor)
        else:
            print('ERROR: strategy ' + strategy + ' not implemented')
        return q_tensor, Ts

