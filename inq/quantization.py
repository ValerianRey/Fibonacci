from inq.fib_util import *
from examples.mnist_models import *
from examples.print_util import count_out
from examples.supported_modules import supported_modules
import torch

ACC_BITS = 32


def calc_qmin_qmax(num_bits=8, negative=False, fib=False):
    if negative:
        qmin = - 2 ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** num_bits - 1

    if fib:
        qmin = 0
        qmax = (qmax + fib_code_int_down(qmax, num_bits=num_bits)) // 2  # We do that to not induce a bias by the choice of qmax
    return qmin, qmax


def calc_scale_zero_point(low_val, high_val, num_bits=8, fib=False):
    qmin, qmax = calc_qmin_qmax(num_bits=num_bits, fib=fib)
    scale = (high_val - low_val) / (qmax - qmin)

    zp = int((qmin - low_val / scale).round())
    # The need to clamp zp depends on how we handle it. For example in a linear layer zp_w is only multiplied once, so it depends on what
    # piece of hardware we use to make that multiplication (is it important to have it on uint8, or is int32 ok, or even float32?)
    # For a conv layer, more multiplications of zp_w are required, so it might be interesting in the future to fibonacci encode the zp_w_kernel
    # and in that case it would probably be better to have zp_w already clamped.
    # zp_x never needs to be clamped because anyway the zp_x value will be added to a int32 accumulator, which will be clamped before needing to
    # be in uint8 format.
    if zp < qmin:
        print(Color.RED + 'zp less than qmin' + Color.END)
        zp = qmin
    elif zp > qmax:
        print(Color.RED + 'zp more than qmax' + Color.END)
        zp = qmax

    return scale, zp  # zero_point needs to be int


# Can be very long to compute for high number of bits
def get_mult_shift(val, num_mult_bits=8, num_shift_bits=32):
    best_mult = 1
    best_shift = 0
    best_diff = abs(val - best_mult)
    for mult in range(1, 2 ** num_mult_bits):
        for shift in range(0, num_shift_bits):
            s_val = val * (2 ** shift)
            if abs(s_val - mult) < best_diff:
                best_diff = abs(s_val - mult)
                best_mult = mult
                best_shift = shift
    return best_mult, best_shift


def quantize_tensor(x, scale, zp):
    return (zp + (x / scale)).round()


def dequantize_tensor(q_x, scale, zp):
    return scale * (q_x.float() - zp)


def compute_quantized_layer(layer, low_val, high_val, scale_x, num_bits=8, fib=False):
    low_val_w, high_val_w = layer.weight.data.min(), layer.weight.data.max()
    scale_w, zp_w = calc_scale_zero_point(low_val_w, high_val_w, num_bits=num_bits, fib=fib)
    q_w = quantize_tensor(layer.weight.data, scale_w, zp_w)
    q_b = quantize_tensor(layer.bias.data, scale_w * scale_x, 0)

    # Compute scale and zero_point from min and max statistics
    scale_next, zp_next = calc_scale_zero_point(low_val=low_val, high_val=high_val, num_bits=num_bits, fib=False)
    combined_scale = scale_x.item() * scale_w.item() / scale_next.item()

    best_mult, best_shift = get_mult_shift(combined_scale, num_bits, ACC_BITS)

    # Fibonacci encode the weights (this is very under efficient due to apply_ not working on cuda)
    if fib:
        q_w = q_w.int().cpu().apply_(lambda x: fib_code_int(x, num_bits=num_bits)).float().cuda()

    return q_w, q_b, best_shift, best_mult, zp_next, scale_next, zp_w, combined_scale


def compute_qmodel(model, stats, num_bits=8, fib=False):
    # Copy the model into qmodel (and its device)
    device = torch.device('cuda')
    qmodel = type(model)()  # get a new instance
    qmodel.load_state_dict(model.state_dict())  # copy weights and stuff
    qmodel.to(device)

    # Choose which stat to use
    low_key = 'min'
    high_key = 'max'

    layers = []
    stat_names = []
    i = 0
    for layer in qmodel.seq:
        if type(layer) in supported_modules:
            if len(layers) > 0:  # there is a shift of 1 in the name: for layer conv1 we use stats['conv2'] for example for the original MNIST net.
                stat_names.append(repr(i))
            layers.append(layer)
            i += 1
    stat_names.append('out')  # This stat is actually not used since we'll hardcode the mult and shift of last layer to 1 and 0

    # Initialization
    qmodel.low_val_input = stats['0'][low_key]
    qmodel.high_val_input = stats['0'][high_key]
    scale, zp = calc_scale_zero_point(low_val=qmodel.low_val_input, high_val=qmodel.high_val_input, num_bits=num_bits)

    for name, layer in zip(stat_names, layers):
        stat = stats[name]

        low_val = stat[low_key]
        high_val = stat[high_key]
        q_w, q_b, shift, mult, zp_next, scale_next, zp_w, combined_scale = \
            compute_quantized_layer(layer, low_val, high_val, scale, num_bits=num_bits, fib=fib)

        layer.weight.data = q_w
        layer.bias.data = q_b
        layer.zp = zp
        layer.zp_w = zp_w
        if name == 'out':
            layer.shift = 0
            layer.mult = 1
            layer.zp_next = 0
        else:
            layer.shift = shift
            layer.mult = mult
            layer.zp_next = zp_next

        if type(layer) == nn.Conv2d:
            # Use only 1 out channel since anyway all kernels are the same
            layer.zp_w_kernel = nn.Conv2d(layer.in_channels, 1, layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                          dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)
            layer.zp_w_kernel.weight.data.fill_(zp_w)
            layer.zp_w_kernel.weight.data = layer.zp_w_kernel.weight.data.cuda()  # TODO: clean that
            layer.unbiased_layer = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                             dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)
            layer.unbiased_layer.weight.data = layer.weight.data
            layer.is_fc = False

        if type(layer) == nn.Linear:
            layer.zp_w_kernel = nn.Linear(layer.in_features, layer.out_features, bias=False)
            layer.zp_w_kernel.weight.data.fill_(zp_w)
            layer.zp_w_kernel.weight.data = layer.zp_w_kernel.weight.data.cuda()  # TODO: clean that
            layer.unbiased_layer = nn.Linear(layer.in_features, layer.out_features, bias=False)
            layer.unbiased_layer.weight.data = layer.weight.data
            layer.is_fc = True

        scale = scale_next
        zp = zp_next

    return qmodel


def qlayer_forward(q_x, layer, layer_stats=None, use_mean=False):
    log = False
    if log:
        print(Color.YELLOW + "x_min=" + repr(q_x.min().item()) + ", x_max=" + repr(q_x.max().item()) + Color.END)

    part1 = layer(q_x)

    if layer.is_fc:  # For linear layers only
        q_x_sum = torch.sum(q_x, dim=1)
        part2 = torch.unsqueeze(q_x_sum * layer.zp_w, dim=1)
    else:
        part2 = layer.zp_w_kernel(q_x)  # Apply the convolution with the zp_w_kernel (way less computations than with a conv layer since it only has 1 out channel)

    if use_mean:
        part3 = layer.part3
        part4 = layer.part4
    else:
        zp_x_vec = torch.zeros_like(q_x).fill_(layer.zp)
        part3 = layer.unbiased_layer(zp_x_vec)
        part4 = layer.zp_w_kernel(zp_x_vec)

    result = part1 - part2 - part3 + part4

    if layer_stats is not None:
        layer_stats['part3'].append(torch.unsqueeze(torch.mean(part3, dim=0), dim=0))
        layer_stats['part4'].append(torch.unsqueeze(torch.mean(part4, dim=0), dim=0))

    if log:
        print(Color.GRAY + 'result_min=' + repr(result.min().item()) + ', result_max=' + repr(result.max().item()) + Color.END)

    # Rescale the result so that: we get rid of the scaling of this layer, and we scale it properly for the next layer
    # We could use int instead of long for 8 bits (no risk of overflowing the int32 range)
    output = ((layer.mult * result.long()) >> layer.shift).float() + layer.zp_next

    if log:
        print(Color.GRAY + 'output_min=' + repr(output.min().item()) + ', output_max=' + repr(output.max().item()) + Color.END)
    return output  # output is an int number stored as float


def qmodel_forward(qmodel, x, num_bits=8, layers_stats=None):
    # Quantise before inputting into incoming layers (no dropout since this is never used for training anyway)
    print_clamped_values = False
    use_mean = True
    if print_clamped_values:
        print()

    input_qmin, input_qmax = calc_qmin_qmax(num_bits)

    scale_x, zp_x = calc_scale_zero_point(qmodel.low_val_input, qmodel.high_val_input, num_bits=num_bits, fib=False)
    x = quantize_tensor(x, scale_x, zp_x)
    too_low_sum = 0
    too_high_sum = 0

    i = 0
    for layer in qmodel.seq:
        if type(layer) in supported_modules:
            too_low, too_high = count_out(x, input_qmin, input_qmax, log=print_clamped_values)
            too_low_sum += too_low
            too_high_sum += too_high
            x = torch.clamp(x, input_qmin, input_qmax)  # Clamp to be sure that we stay within the uint8 range
            if layers_stats is not None:
                x = qlayer_forward(x, layer, layers_stats[i])
            else:
                x = qlayer_forward(x, layer, use_mean=use_mean)
            i += 1
        else:
            x = layer(x)

    return x


def enhance_qmodel(qmodel, layers_means):
    i = 0
    for layer in qmodel.seq:  # Only iterate over the main modules and not the modules contained in those
        if type(layer) in supported_modules:
            layer.part3 = layers_means[i]['part3']
            layer.part4 = layers_means[i]['part4']
            i += 1

    return qmodel  # Works in place but still returns qmodel



