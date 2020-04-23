from inq.fib_util import *
from examples.mnist_models import *


ACC_BITS = 32


def calc_qmin_qmax(num_bits=8, negative=False):
    if negative:
        qmin = - 2 ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** num_bits - 1
    return qmin, qmax


def calc_scale_zero_point(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin, qmax = calc_qmin_qmax(num_bits=num_bits)
    scale = (max_val - min_val) / (qmax - qmin)

    #zero_point = max(min(int(qmin - min_val / scale), qmax), qmin)
    zero_point = int((qmin - min_val / scale).round())
    return scale, zero_point  # zero_point needs to be int


def get_mult_shift(val, num_mult_bits=8, num_shift_bits=32):
    best_diff = 1000000000000000000000000000000000000
    best_mult = 1
    best_shift = 0
    for mult in range(1, 2 ** num_mult_bits):
        for shift in range(0, num_shift_bits):
            s_val = val * (2 ** shift)
            if abs(s_val - mult) < best_diff:
                best_diff = abs(s_val - mult)
                best_mult = mult
                best_shift = shift

    return best_mult, best_shift


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    scale, zero_point = calc_scale_zero_point(min_val, max_val, num_bits)
    q_x = (zero_point + x / scale).round()

    return q_x, scale, zero_point  # q_x is an integer number stored as a float


def quantize_bias(b, scale):
    q_b = (b / scale).round()
    return q_b


def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x.float() - zero_point)


def compute_quantized_layer(layer, stat, scale_x, num_bits=8, fibonacci_encode=False):
    # Quantize the layer: find a an appropriate scale and zp for W, then quantize W with it and B with the same scale as W (but no zp)
    q_w, scale_w, zp_w = quantize_tensor(layer.weight.data, num_bits=num_bits)
    # Maybe b should be quantized with scale=scale_w * scale_x TODO: check that
    q_b = quantize_bias(layer.bias.data, scale_w * scale_x)

    # Compute scale and zero_point from min and max statistics
    scale_next, zero_point_next = calc_scale_zero_point(min_val=stat['min'], max_val=stat['max'], num_bits=num_bits)
    #zero_point_next = 0  # TODO: verify that this is not wrong
    combined_scale = scale_x.item() * scale_w.item() / scale_next.item()
    best_mult, best_shift = get_mult_shift(combined_scale, num_bits, ACC_BITS)
    # q_w = q_w  - zp_w  # Comment this so that zp_w is removed after the multiplications so that all computations are made with positive integers (easily fib encodable)

    # Fibonacci encode the weights (this is very under efficient due to apply_ not working on cuda)
    if fibonacci_encode:
        q_w = q_w.int().cpu()
        q_w.apply_(fib_code_int)
        q_w = q_w.float().cuda()
        # zp_w = fib_code_int(zp_w)  # Quantize zp_w as well to only have fibonacci encoded multiplications
        # Note that this might reduce a lot the accuracy while being very very negligible in terms of computation time gained (for the Linear layers at least,
        # because there might be a shortcut that makes it require only 1 multiplication

    return q_w, q_b, best_shift, best_mult, zero_point_next, scale_next, zp_w, combined_scale


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
    # TODO: verify that it is right to use the second line and not the first
    #stat_names.append(stat_names[-1])  # we use the stats of the last layer twice
    stat_names.append('out')

    # for name, layer in qmodel.named_modules():
    #     if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
    #         stat_names.append(name)
    #         layers.append(layer)

    for name, layer in zip(stat_names, layers):
        stat = stats[name]

        q_w, q_b, best_shift, best_mult, zp_next, scale_next, zp_w, combined_scale = compute_quantized_layer(layer, stat, scale, num_bits=num_bits, fibonacci_encode=fibonacci_encode)

        layer.weight.data = q_w
        layer.bias.data = q_b
        layer.shift = best_shift
        layer.mult = best_mult
        layer.zp = zp
        layer.combined_scale = combined_scale  # Just used for testing, this is included already inside of mult and shift
        layer.zp_next = zp_next
        layer.zp_w = zp_w

        if type(layer) == nn.Conv2d:
            # layer.zp_w_kernel = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding,
            #                               dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)

            # Use only 1 out channel since anyway all kernels are the same
            layer.zp_w_kernel = nn.Conv2d(layer.in_channels, 1, layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                          dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)
            layer.zp_w_kernel.weight.data.fill_(zp_w)
            layer.zp_w_kernel.weight.data = layer.zp_w_kernel.weight.data.cuda()  # TODO: clean that
            layer.unbiased_layer = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                             dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)
            layer.unbiased_layer.weight.data = layer.weight.data
            layer.sum_dim = 2  # TODO: make that generic

        if type(layer) == nn.Linear:
            layer.zp_w_kernel = nn.Linear(layer.in_features, layer.out_features, bias=False)
            layer.zp_w_kernel.weight.data.fill_(zp_w)
            layer.zp_w_kernel.weight.data = layer.zp_w_kernel.weight.data.cuda()  # TODO: clean that
            layer.unbiased_layer = nn.Linear(layer.in_features, layer.out_features, bias=False)
            layer.unbiased_layer.weight.data = layer.weight.data
            layer.sum_dim = 1  # TODO: make that generic

        scale = scale_next
        zp = zp_next

    return qmodel


def qlayer_forward(x, layer, layer_stats=None, use_mean=False):
    log = False
    if log:
        print(Color.GRAY + "x_min=" + repr(x.min().item()) + ", x_max=" + repr(x.max().item()) + Color.END)

    q_x = x
    zp_x_vec = torch.zeros_like(x).fill_(layer.zp)

    part1 = layer(q_x)

    if layer.sum_dim == 1:  # For linear layers only
        q_x_sum = torch.sum(q_x, dim=layer.sum_dim)
        part2 = torch.unsqueeze(q_x_sum * layer.zp_w, dim=1)
    else:
        part2 = layer.zp_w_kernel(q_x)  # Apply the convolution with the zp_w_kernel (way less computations than with a conv layer since it only has 1 out channel)

    if use_mean:
        part3 = layer.part3
        part4 = layer.part4
    else:
        part3 = layer.unbiased_layer(zp_x_vec)
        part4 = layer.zp_w_kernel(zp_x_vec)

    result = part1 - part2 - part3 + part4

    if layer_stats is not None:
        layer_stats['part1'].append(torch.unsqueeze(torch.mean(part1, dim=0), dim=0))
        layer_stats['part2'].append(torch.unsqueeze(torch.mean(part2, dim=0), dim=0))
        layer_stats['part3'].append(torch.unsqueeze(torch.mean(part3, dim=0), dim=0))
        layer_stats['part4'].append(torch.unsqueeze(torch.mean(part4, dim=0), dim=0))

    if log:
        print(Color.GRAY + 'result_min=' + repr(result.min().item()) + ', result_max=' + repr(result.max().item()) + Color.END)

    # Rescale the result so that: we get rid of the scaling of this layer, and we scale it properly for the next layer
    output = ((layer.mult * result.int()) >> layer.shift).float() + layer.zp_next

    if log:
        print(Color.GRAY + 'output_min=' + repr(output.min().item()) + ', output_max=' + repr(output.max().item()) + Color.END)
    return output  # result_scaled_for_next_layer is an int32 number


def qmodel_forward(qmodel, x, stats, num_bits=8, layers_stats=None):
    # Quantise before inputting into incoming layers (no dropout since this is never used for training anyway)

    # The first line ensures that all x are quantized with the same scale / zp
    x, _, _ = quantize_tensor(x, num_bits=num_bits, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])
    # x, _, _ = quantize_tensor(x, num_bits=num_bits)

    use_mean = True

    if layers_stats is not None:
        x = qlayer_forward(x, qmodel.conv1, layers_stats[0])
    else:
        x = qlayer_forward(x, qmodel.conv1, use_mean=use_mean)
    x = F.relu(x)

    if layers_stats is not None:
        x = qlayer_forward(x, qmodel.conv2, layers_stats[1])
    else:
        x = qlayer_forward(x, qmodel.conv2, use_mean=use_mean)
    x = F.max_pool2d(x, 2)
    x = torch.flatten(x, 1)

    if layers_stats is not None:
        x = qlayer_forward(x, qmodel.fc1, layers_stats[2])
    else:
        x = qlayer_forward(x, qmodel.fc1, use_mean=use_mean)
    x = F.relu(x)

    if layers_stats is not None:
        x = qlayer_forward(x, qmodel.fc2, layers_stats[3])
    else:
        x = qlayer_forward(x, qmodel.fc2, use_mean=use_mean)

    return x


# For now this works only on the baseline network (not generic)
def enhance_qmodel(qmodel, layers_means):
    qmodel.conv1.part1 = layers_means[0]['part1']  # part 1 is actually unused (otherwise everything would be constant)
    qmodel.conv1.part2 = layers_means[0]['part2']
    qmodel.conv1.part3 = layers_means[0]['part3']
    qmodel.conv1.part4 = layers_means[0]['part4']

    qmodel.conv2.part1 = layers_means[1]['part1']
    qmodel.conv2.part2 = layers_means[1]['part2']
    qmodel.conv2.part3 = layers_means[1]['part3']
    qmodel.conv2.part4 = layers_means[1]['part4']

    qmodel.fc1.part1 = layers_means[2]['part1']
    qmodel.fc1.part2 = layers_means[2]['part2']
    qmodel.fc1.part3 = layers_means[2]['part3']
    qmodel.fc1.part4 = layers_means[2]['part4']

    qmodel.fc2.part1 = layers_means[3]['part1']
    qmodel.fc2.part2 = layers_means[3]['part2']
    qmodel.fc2.part3 = layers_means[3]['part3']
    qmodel.fc2.part4 = layers_means[3]['part4']

    return qmodel  # Works in place but still returns qmodel
