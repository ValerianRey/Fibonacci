import torch.nn.functional as F
import torch


def binary_int(num):
    code = bin(num)[2:]
    if len(code) > 32:
        print("ERROR: num needs more than 32 bits")
        return '0' * 32
    else:
        return '0' * (32-len(code)) + code


def int_from_bin(code):
    return int(code, 2)


# Gives the Fibonacci-valid int-32 number that is the closest (down) to our int-32 number 'num'
def fib_code_int_down(num):
    code = list(binary_int(num))
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


# Gives the Fibonacci-valid int-32 number that is the closest (up) to our int-32 number 'num'
def fib_code_int_up(num):
    code = list(binary_int(num))
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
                        break;

                i -= 1
                while i < len(code):
                    code[i] = '0'  # Remove the problem (and all subsequent problems on the right)
                    i += 1
                break

        else:
            count = 0

    code = ''.join(code)
    return int_from_bin(code)


# Gives the best Fibonacci-valid int-32 approximation for 'num'
def fib_code_int(num):
    down = fib_code_int_down(num)
    up = fib_code_int_up(num)
    dist_down = abs(num - down)
    dist_up = abs(num - up)
    if dist_down < dist_up:
        return down
    else:
        return up


def calc_scale_zero_point(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = int(max(min(qmin - min_val / scale, qmax), qmin))

    return scale, zero_point


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calc_scale_zero_point(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    # Todo: make cuda not hard coded here
    return q_x.cuda(), scale, zero_point


def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x.float() - zero_point)


def quantize_layer(x, layer, stat, scale_x, zp_x):
    # for both conv and linear layers

    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    # quantise weights, activations are already quantised
    w, w_scale, w_zero_point = quantize_tensor(layer.weight.data)
    b, b_scale, b_zero_point = quantize_tensor(layer.bias.data)

    layer.weight.data = w.float()
    layer.bias.data = b.float()

    # This is Quantisation Artihmetic
    scale_w = w_scale
    zp_w = w_zero_point
    scale_b = b_scale
    zp_b = b_zero_point

    scale_next, zero_point_next = calc_scale_zero_point(min_val=stat['min'], max_val=stat['max'])

    x = scale_x * (x.float() - zp_x)  # Dequantize x
    layer.weight.data = scale_w * (layer.weight.data - zp_w)  # Dequantize the layer weights
    layer.bias.data = scale_b * (layer.bias.data - zp_b)  # Dequantize the layer biases

    # All int computation
    x = (layer(x) / scale_next) + zero_point_next  # Forward pass the layer and quantize the result

    # Reset weights for next forward pass
    layer.weight.data = W
    layer.bias.data = B

    return x, scale_next, zero_point_next


# Get Min and max of x tensor, and stores it
def update_stats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)
    batch_size = max_val.shape[0]

    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": batch_size}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += batch_size


# Reworked Forward Pass to access activation Stats through update_stats function
def gather_activation_stats(model, x, stats):
    update_stats(x.clone().view(x.shape[0], -1), stats, 'conv1')
    x = model.conv1(x)
    x = F.relu(x)

    update_stats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    x = model.conv2(x)
    x = F.max_pool2d(x, 2)
    x = model.dropout1(x)
    x = torch.flatten(x, 1)

    update_stats(x, stats, 'fc1')
    x = model.fc1(x)
    x = F.relu(x)
    x = model.dropout2(x)

    update_stats(x, stats, 'fc2')


# Entry function to get stats of all functions.
def gather_stats(model, test_loader):
    device = 'cuda'

    model.eval()
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            gather_activation_stats(model, data, stats)

    final_stats = {}
    for key, value in stats.items():
        final_stats[key] = {"max": value["max"] / value["total"], "min": value["min"] / value["total"]}
    return final_stats


def quant_forward(model, x, stats):
    # Quantise before inputting into incoming layers
    x, scale, zero_point = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])

    x, scale_next, zero_point_next = quantize_layer(x, model.conv1, stats['conv2'], scale, zero_point)
    x = F.relu(x)

    x, scale_next, zero_point_next = quantize_layer(x, model.conv2, stats['fc1'], scale_next, zero_point_next)
    x = F.max_pool2d(x, 2)
    x = model.dropout1(x)
    x = torch.flatten(x, 1)

    x, scale_next, zero_point_next = quantize_layer(x, model.fc1, stats['fc2'], scale_next, zero_point_next)
    x = F.relu(x)
    x = model.dropout2(x)

    # Back to dequant for final layer
    x = dequantize_tensor(x, scale_next, zero_point_next)

    x = model.fc2(x)

    return F.log_softmax(x, dim=1)


def scale_tensor(x, bits=8):
    qmax = 2 ** bits - 1
    qmin = 0
    with torch.no_grad():
        min_val = torch.min(x).item()
        max_val = torch.max(x).item()
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(max(min(qmin - min_val / scale, qmax), qmin))
        q_x = zero_point + x / scale
        q_x.clamp_(qmin, qmax).round_()
        q_x = q_x.round()

    return q_x


