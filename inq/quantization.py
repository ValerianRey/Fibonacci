from inq.fib_util import *
from examples.mnist_models import *


def calc_scale_zero_point(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = int(max(min(qmin - min_val / scale, qmax), qmin))

    return scale, zero_point


def get_mult_shift(val, num_mult_bits=8, num_shift_bits=8):  # The default 32 bits seems to lead to bugs
    best_diff = 1000000000000000000000000000000000000
    for mult in range(1, 2 ** num_mult_bits):
        for shift in range(1, num_shift_bits):
            s_val = val * (2 ** shift)
            if abs(s_val - mult) < best_diff:
                best_diff = abs(s_val - mult)
                best_mult = mult
                best_shift = shift

    return best_mult, best_shift


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calc_scale_zero_point(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().int()

    return q_x, scale, zero_point


def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x.float() - zero_point)


def compute_quantized_layer(layer, stat, scale_x, num_bits=8, fibonacci_encode=False):
    # Copy the layer parameters
    W = layer.weight.data
    B = layer.bias.data

    # quantise weights
    W, scale_w, zp_w = quantize_tensor(W, num_bits=num_bits)
    B, scale_b, zp_b = quantize_tensor(B, num_bits=num_bits)

    # Turn the layer into float type (even though the numbers are actually integers)
    W = W.float()
    B = B.float()

    # Compute scale and zero_point from min and max statistics
    scale_next, zero_point_next = calc_scale_zero_point(min_val=stat['min'], max_val=stat['max'], num_bits=num_bits)
    combined_scale = scale_x.item() * scale_w.item() / scale_next.item()
    best_mult, best_shift = get_mult_shift(combined_scale, num_bits, num_bits)
    W = best_mult * (W - zp_w)
    B = best_mult * (B - zp_b)

    print(W)

    # Fibonacci encode the weights (this is very under efficient due to apply_ not working on cuda)
    if fibonacci_encode:
        W = W.char().cpu()
        W.apply_(fib_code_int)
        W = W.float().cuda()

    return W, B, best_shift, zero_point_next, scale_next


def compute_qmodel(model, stats, num_bits=8, fibonacci_encode=False):
    # Copy the model into qmodel (and its device)
    device = model.conv1.weight.device
    qmodel = type(model)()  # get a new instance
    qmodel.load_state_dict(model.state_dict())  # copy weights and stuff
    qmodel.to(device)

    # Initialization
    scale, zp = calc_scale_zero_point(min_val=stats['conv1']['min'], max_val=stats['conv1']['max'], num_bits=num_bits)

    layers = []
    stat_names = []

    for name, layer in qmodel.named_modules():
        if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
            if len(layers) > 0:  # there is a shift of 1 in the name: for layer conv1 we use stats['conv2'] for example for the original MNIST net.
                stat_names.append(name)
            layers.append(layer)

    layers = layers[:-1]  # we do not quantise the last layer

    for name, layer in zip(stat_names, layers):
        stat = stats[name]
        W, B, best_shift, zp_next, scale_next = compute_quantized_layer(layer, stat, scale, num_bits=num_bits, fibonacci_encode=fibonacci_encode)

        layer.weight.data = W
        layer.bias.data = B
        layer.best_shift = best_shift
        layer.zp = zp
        layer.zp_next = zp_next
        layer.scale_next = scale_next

        scale = layer.scale_next
        zp = layer.zp_next

    return qmodel


def qlayer_forward(x, layer):
    x = x.float()
    x = x - layer.zp
    # All int computation
    x = ((layer(x) >> layer.best_shift).round().int() + layer.zp_next).float()

    return x


def qmodel_forward(qmodel, x, stats, num_bits=8):
    # Quantise before inputting into incoming layers
    x, _, _ = quantize_tensor(x, num_bits=num_bits, min_val=stats['conv1']['min'],
                                           max_val=stats['conv1']['max'])

    layer = qmodel.conv1
    x = qlayer_forward(x, layer)
    x = F.relu(x)

    layer = qmodel.conv2
    x = qlayer_forward(x, layer)
    x = F.max_pool2d(x, 2)
    x = qmodel.dropout1(x)
    x = torch.flatten(x, 1)

    layer = qmodel.fc1
    x = qlayer_forward(x, layer)
    x = F.relu(x)
    x = qmodel.dropout2(x)

    # Back to dequant for final layer
    x = dequantize_tensor(x, layer.scale_next, layer.zp_next)

    x = qmodel.fc2(x)

    return F.log_softmax(x, dim=1)

