from inq.fib_util import *
from examples.mnist_models import *
from examples.print_util import Color, count_out
from examples.supported_modules import supported_modules
from examples.supported_modules import batch_norm_modules
import torch
import copy

ACC_BITS = 32


def calc_qmin_qmax(bits=8, negative=False, fib=False):
    if negative:
        qmin = - 2 ** (bits - 1)
        qmax = 2. ** (bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** bits - 1

    if fib:
        qmin = 0
        qmax = (qmax + fib_code_int_down(qmax, bits=bits)) // 2  # We do that to not induce a bias by the choice of qmax
    return qmin, qmax


def calc_scale_zero_point(low_val, high_val, bits=8, fib=False):
    qmin, qmax = calc_qmin_qmax(bits=bits, fib=fib)
    scale = (high_val - low_val) / (qmax - qmin)
    zp = round((qmin - low_val / scale))
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
def get_mult_shift(val, mult_bits=8, shift_bits=32):
    best_mult = 1
    best_shift = 0
    best_diff = abs(val - best_mult)
    for mult in range(1, 2 ** mult_bits):
        for shift in range(0, shift_bits):
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


def compute_quantized_layer(layer, scale_x, scale_x_next, bits=8, fib=False, proportion=1.):
    low_val_w, high_val_w = layer.weight.data.min().item(), layer.weight.data.max().item()
    scale_w, zp_w = calc_scale_zero_point(low_val_w, high_val_w, bits=bits, fib=fib)
    q_w = quantize_tensor(layer.weight.data, scale_w, zp_w)
    if layer.bias is not None:
        q_b = quantize_tensor(layer.bias.data, scale_w * scale_x, 0)
    else:
        q_b = None

    combined_scale = scale_x * scale_w / scale_x_next
    best_mult, best_shift = get_mult_shift(combined_scale, bits, ACC_BITS)

    # Fibonacci encode the weights (this is very under efficient due to apply_ not working on cuda)
    if fib:
        q_w, Ts = fib_quantize_tensor(q_w, proportion, bits=bits)
    else:
        Ts = torch.ones_like(q_w)

    return q_w, q_b, best_shift, best_mult, zp_w, scale_w, scale_x, combined_scale, Ts


def compute_qmodel(model, stats, optimizer, bits=8, fib=False, proportion=1.):
    # Copy the model into qmodel (and its device)
    qmodel = copy.deepcopy(model)

    # Choose which stat to use
    low_key = 'min'
    high_key = 'max'

    layers = []
    stat_names = []
    indices = []  # Contains the id of the parameters that will be quantized (as stored in the optimizer)

    with torch.no_grad():
        for j in range(len(qmodel.seq)):
            # Completely deactivate BatchNorm modules.
            # The normalization should already be made by our quantization pipeline, and the affine learning should be set to false in the training.
            if type(qmodel.seq[j]) in batch_norm_modules:
                print('Disabling batch norm module')
                qmodel.seq[j] = nn.Identity()

        i = 0
        idx = 0
        for layer in qmodel.seq:
            if type(layer) in supported_modules:
                for name, _ in layer.named_parameters():
                    if name == 'weight':
                        indices.append(idx)
                    idx += 1
                if len(layers) > 0:  # there is a shift of 1 in the name: for layer conv1 we use stats['conv2'] for example for the original MNIST net.
                    stat_names.append(repr(i))
                layers.append(layer)
                i += 1

            else:
                for _ in layer.parameters():
                    idx += 1

        stat_names.append('none')  # Dummy stat name to indicate that we are at the last layer and we do not actually need the stat

        # Initialization
        qmodel.register_parameter('low_val_input', nn.Parameter(torch.tensor(stats['0'][low_key]), requires_grad=False))
        qmodel.register_parameter('high_val_input', nn.Parameter(torch.tensor(stats['0'][high_key]), requires_grad=False))
        scale_x, zp_x = calc_scale_zero_point(low_val=qmodel.low_val_input.item(), high_val=qmodel.high_val_input.item(), bits=bits)

        assert(len(stat_names) == len(layers) and len(layers) == len(indices))
        # for name, layer, fib_layer in zip(stat_names, layers, fib_layers):
        for name, layer, idx in zip(stat_names, layers, indices):
            if name == 'none':
                scale_x_next, zp_x_next = 1.0, 0
            else:
                scale_x_next, zp_x_next = calc_scale_zero_point(low_val=stats[name][low_key], high_val=stats[name][high_key], bits=bits, fib=False)
            q_w, q_b, shift, mult, zp_w, scale_w, scale_x, combined_scale, Ts = \
                compute_quantized_layer(layer, scale_x, scale_x_next, bits=bits, fib=fib, proportion=proportion)  # fib=(fib and fib_layer)
            optimizer.param_groups[0]['Ts'][idx] = Ts

            layer.weight.data = q_w
            if layer.bias is not None:
                layer.bias.data = q_b
            layer.register_parameter('zp_x', torch.nn.Parameter(data=torch.tensor(zp_x, device='cuda'), requires_grad=False))
            layer.register_parameter('zp_w', torch.nn.Parameter(data=torch.tensor(zp_w, device='cuda'), requires_grad=False))

            layer.register_parameter('scale_b', torch.nn.Parameter(data=torch.tensor(scale_x * scale_w, device='cuda'), requires_grad=False))
            layer.register_parameter('scale_w', torch.nn.Parameter(data=torch.tensor(scale_w, device='cuda'), requires_grad=False))
            if name == 'none':
                layer.register_parameter('shift', torch.nn.Parameter(data=torch.tensor(0, device='cuda'), requires_grad=False))
                layer.register_parameter('mult', torch.nn.Parameter(data=torch.tensor(1, device='cuda'), requires_grad=False))
                layer.register_parameter('zp_x_next', torch.nn.Parameter(data=torch.tensor(0, device='cuda'), requires_grad=False))
            else:
                layer.register_parameter('shift', torch.nn.Parameter(data=torch.tensor(shift, device='cuda'), requires_grad=False))
                layer.register_parameter('mult', torch.nn.Parameter(data=torch.tensor(mult, device='cuda'), requires_grad=False))
                layer.register_parameter('zp_x_next', torch.nn.Parameter(data=torch.tensor(zp_x_next, device='cuda'), requires_grad=False))

            if type(layer) in supported_modules:
                if type(layer) == nn.Conv2d:
                    # Use only 1 out channel since anyway all kernels are the same
                    layer.zp_w_kernel = nn.Conv2d(layer.in_channels, 1, layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                                  dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)
                    layer.zp_w_kernel.weight.data.fill_(zp_w)
                    layer.zp_w_kernel.weight.data = layer.zp_w_kernel.weight.data.cuda()
                    layer.unbiased_layer = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                                     dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)
                    layer.unbiased_layer.weight.data = layer.weight.data

                if type(layer) == nn.Linear:
                    layer.zp_w_kernel = nn.Linear(layer.in_features, layer.out_features, bias=False)
                    layer.zp_w_kernel.weight.data.fill_(zp_w)
                    layer.zp_w_kernel.weight.data = layer.zp_w_kernel.weight.data.cuda()
                    layer.unbiased_layer = nn.Linear(layer.in_features, layer.out_features, bias=False)
                    layer.unbiased_layer.weight.data = layer.weight.data

                for p in layer.zp_w_kernel.parameters():
                    p.requires_grad = False
                for p in layer.unbiased_layer.parameters():
                    p.requires_grad = False

            scale_x = scale_x_next
            zp_x = zp_x_next

    return qmodel


# This function descales an int (+fib) quantized network and puts it in its original form
# It also uses the original model before int (+fib) quantization to find the exact expected structure
def update_model(model, qmodel):
    with torch.no_grad():
        for i, qlayer in enumerate(qmodel.seq):
            if type(qlayer) in supported_modules:  # Only these layers have been modified
                model.seq[i].weight.data = dequantize_tensor(qlayer.weight.data, qlayer.scale_w, qlayer.zp_w)
                if model.seq[i].bias is not None:
                    model.seq[i].bias.data = dequantize_tensor(qlayer.bias.data, qlayer.scale_b, 0)


def update_qmodel(qmodel, model):
    with torch.no_grad():
        for i, layer in enumerate(model.seq):
            if type(layer) in supported_modules:  # Only these layers have been modified
                qmodel.seq[i].weight.data = quantize_tensor(model.seq[i].weight.data, qmodel.seq[i].scale_w, qmodel.seq[i].zp_w)
                if qmodel.seq[i].bias is not None:
                    qmodel.seq[i].bias.data = quantize_tensor(model.seq[i].bias.data, qmodel.seq[i].scale_b, 0)
                qmodel.seq[i].unbiased_layer.weight.data = qmodel.seq[i].weight.data
                for p in qmodel.seq[i].unbiased_layer.parameters():
                    p.requires_grad = False


def increase_fib_proportion(qmodel, optimizer, bits, proportion):
    with torch.no_grad():
        idx = 0
        for layer in qmodel.seq:
            if type(layer) in supported_modules:
                for name, param in layer.named_parameters():
                    if name == 'weight':
                        param.data, optimizer.param_groups[0]['Ts'][idx] = fib_quantize_tensor(param.data, proportion, bits=bits)
                    if name == 'weight' or name == 'bias':
                        idx += 1
                layer.unbiased_layer.weight.data = layer.weight.data

            else:
                for _ in layer.parameters():
                    idx += 1


def qlayer_forward(q_x, layer, layer_stats=None, use_mean=False):
    gathering_stats = layer_stats is not None
    log = False
    if log and not gathering_stats:
        print(Color.YELLOW + "x_min=" + repr(q_x.min().item()) + ", x_max=" + repr(q_x.max().item()) + Color.END)

    part1 = layer(q_x)

    if type(layer) == nn.Linear:  # For linear layers only
        q_x_sum = torch.sum(q_x, dim=1)
        part2 = torch.unsqueeze(q_x_sum * layer.zp_w, dim=1)
    else:
        part2 = layer.zp_w_kernel(q_x)  # Apply the convolution with the zp_w_kernel (way less computations than with a conv layer since it only has 1 out channel)

    if use_mean:
        part3 = layer.part3
        part4 = layer.part4
    else:
        zp_x_vec = torch.zeros_like(q_x).fill_(layer.zp_x)
        part3 = layer.unbiased_layer(zp_x_vec)
        part4 = layer.zp_w_kernel(zp_x_vec)

    result = part1 - part2 - part3 + part4

    if gathering_stats:
        layer_stats['part3'].append(torch.unsqueeze(torch.mean(part3, dim=0), dim=0))
        layer_stats['part4'].append(torch.unsqueeze(torch.mean(part4, dim=0), dim=0))

    if log and not gathering_stats:
        print(Color.GRAY + 'result_min=' + repr(result.min().item()) + ', result_max=' + repr(result.max().item()) + Color.END)

    # Rescale the result so that: we get rid of the scaling of this layer, and we scale it properly for the next layer
    # We could use int instead of long for 8 bits (no risk of overflowing the int32 range)
    output = ((layer.mult * result.long()) >> layer.shift).float() + layer.zp_x_next

    if log and not gathering_stats:
        print(Color.GRAY + 'output_min=' + repr(output.min().item()) + ', output_max=' + repr(output.max().item()) + Color.END)
    return output  # output is an int number stored as float


def qmodel_forward(qmodel, x, bits=8, layers_stats=None):
    # Quantise before inputting into incoming layers (no dropout since this is never used for training anyway)
    gathering_stats = layers_stats is not None
    print_clamped_values = False and not gathering_stats  # Never print the clamped values when collecting stats
    use_mean = True
    if print_clamped_values:
        print()

    input_qmin, input_qmax = calc_qmin_qmax(bits)

    scale_x, zp_x = calc_scale_zero_point(qmodel.low_val_input.item(), qmodel.high_val_input.item(), bits=bits, fib=False)
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
            if gathering_stats:
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
            # layer.part3 = layers_means[i]['part3']
            layer.register_parameter('part3', torch.nn.Parameter(layers_means[i]['part3'], requires_grad=False))
            # layer.part4 = layers_means[i]['part4']
            layer.register_parameter('part4', torch.nn.Parameter(layers_means[i]['part4'], requires_grad=False))
            i += 1

    return qmodel  # Works in place but still returns qmodel



