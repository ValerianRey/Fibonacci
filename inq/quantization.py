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
    return torch.tensor(qmin, device='cuda'), torch.tensor(qmax, device='cuda')


def calc_scale_zero_point(low_val, high_val, bits=8, fib=False):
    qmin, qmax = calc_qmin_qmax(bits=bits, fib=fib)
    if high_val == low_val:
        # In this case the value is constant anyway so we just give an arbitrary value to scale that is not infinity
        scale = 1
    else:
        scale = (high_val - low_val) / (qmax - qmin)

    zp = torch.round((qmin - low_val / scale))
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


def calc_scales_zero_points(low_vals, high_vals, bits=8, fib=False):
    scales = torch.ones_like(low_vals)
    zps = torch.zeros_like(low_vals)

    for i, (low_val, high_val) in enumerate(zip(low_vals, high_vals)):
        scale, zp = calc_scale_zero_point(low_val, high_val, bits=bits, fib=fib)
        scales[i] = scale
        zps[i] = zp

    return scales, zps


# Can be very long to compute for high number of bits
def get_mult_shift_old(val, mult_bits=8, shift_bits=32):
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


def get_mult_shift(val, mult_bits=8, shift_bits=32):
    best_mult = 1
    best_shift = 0
    best_diff = abs(val - best_mult)
    for shift in range(shift_bits):
        s_val = val * (2 ** shift)
        mult = min(max(torch.round(s_val), torch.tensor(1)), 2 ** mult_bits - 1)
        if abs(s_val - mult) < best_diff:
            best_diff = abs(s_val - mult)
            best_mult = mult
            best_shift = shift
    return best_mult, best_shift


def get_mults_shifts(vals, mult_bits=8, shift_bits=32):
    best_mults = torch.ones_like(vals)
    best_shifts = torch.zeros_like(vals)

    for i, val in enumerate(vals):
        best_mult, best_shift = get_mult_shift(val, mult_bits=mult_bits, shift_bits=shift_bits)
        best_mults[i] = best_mult
        best_shifts[i] = best_shift

    return best_mults, best_shifts


def unsqueeze_1d_to_4d(x, dim):
    if dim == 0:
        return x.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    elif dim == 1:
        return x.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    elif dim == 2:
        return x.unsqueeze(0).unsqueeze(1).unsqueeze(3)
    else:
        return x.unsqueeze(0).unsqueeze(1).unsqueeze(2)


def quantize_tensor(x, scale, zp):
    return (zp + (x / scale)).round()


def quantize_tensor_4d(x, scales, zps):
    return (unsqueeze_1d_to_4d(zps, dim=0) + (x / unsqueeze_1d_to_4d(scales, dim=0))).round()


def quantize_4d_tensor_per_channel(x, scales, zps):
    # TODO: use that instead of the above
    zps_4d = unsqueeze_1d_to_4d(zps, dim=0)
    return (torch.einsum("c,cnhw->cnhw", 1/scales, x) + zps_4d).round()


def dequantize_tensor(q_x, scale, zp):
    return scale * (q_x.float() - zp)


def dequantize_4d_tensor_per_channel(q_x, scales, zps):
    zps_4d = unsqueeze_1d_to_4d(zps, dim=0)
    return torch.einsum("c,cnhw->cnhw", scales, q_x.float() - zps_4d)  # n is in_channels


def stats_over_4d_tensor_per_channel(x, stat):
    stats = torch.zeros(x.shape[0], device='cuda')
    for channel in range(x.shape[0]):
        stats[channel] = stat(x[channel])
    return stats


def compute_quantized_layer(layer, scale_x, scale_x_next, proportions=None, step=None, bits=8, fib=False, strategy='quantile'):
    if not type(layer) in supported_modules:
        raise TypeError("compute_quantized_layer not implemented for layer of type {}".format(type(layer).__name__))

    # print("Compute quantized layer of type " + repr(type(layer)))
    if type(layer) == nn.Linear:
        low_vals_w, high_vals_w = layer.weight.data.min().unsqueeze(0), layer.weight.data.max().unsqueeze(0)
    elif type(layer) == nn.Conv2d:
        low_vals_w = stats_over_4d_tensor_per_channel(layer.weight.data, torch.min)
        high_vals_w = stats_over_4d_tensor_per_channel(layer.weight.data, torch.max)

    scales_w, zps_w = calc_scales_zero_points(low_vals_w, high_vals_w, bits=bits, fib=fib)
    if type(layer) == nn.Conv2d:
        q_w = quantize_tensor_4d(layer.weight.data, scales_w, zps_w)
    elif type(layer) == nn.Linear:
        q_w = quantize_tensor(layer.weight.data, scales_w, zps_w)

    if layer.bias is not None:
        q_b = quantize_tensor(layer.bias.data, scales_w * scale_x, torch.tensor([0], device='cuda'))
    else:
        q_b = None

    combined_scales = scale_x * scales_w / scale_x_next
    best_mults, best_shifts = get_mults_shifts(combined_scales, bits, ACC_BITS)

    # Fibonacci encode the weights (this is very under efficient due to apply_ not working on cuda)
    if fib:
        q_w, Ts = fib_quantize_tensor(q_w, proportions, step, bits=bits, strategy=strategy)
    else:
        Ts = torch.ones_like(q_w)

    return q_w, q_b, best_shifts, best_mults, zps_w, scales_w, scale_x, combined_scales, Ts


def compute_qmodel(model, stats, optimizer, proportions=None, step=None, bits=8, fib=False, strategy='quantile'):
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
        qmodel.register_parameter('bits', nn.Parameter(torch.tensor(bits), requires_grad=False))
        qmodel.register_parameter('low_val_input', nn.Parameter(stats['0'][low_key], requires_grad=False))
        qmodel.register_parameter('high_val_input', nn.Parameter(stats['0'][high_key], requires_grad=False))
        scale_x, zp_x = calc_scale_zero_point(low_val=qmodel.low_val_input, high_val=qmodel.high_val_input, bits=bits)

        assert(len(stat_names) == len(layers) and len(layers) == len(indices))
        # for name, layer, fib_layer in zip(stat_names, layers, fib_layers):
        for name, layer, idx in zip(stat_names, layers, indices):
            if name == 'none':
                scale_x_next, zp_x_next = 1.0, 0
            else:
                scale_x_next, zp_x_next = calc_scale_zero_point(low_val=stats[name][low_key], high_val=stats[name][high_key], bits=bits, fib=False)
            q_w, q_b, shifts, mults, zps_w, scales_w, scale_x, combined_scales, Ts = \
                compute_quantized_layer(layer, scale_x, scale_x_next, proportions=proportions, step=step, bits=bits, fib=fib, strategy=strategy)  # fib=(fib and fib_layer)

            optimizer.param_groups[0]['Ts'][idx] = Ts
            layer.weight.data = q_w
            if layer.bias is not None:
                layer.bias.data = q_b
            layer.register_parameter('zp_x', torch.nn.Parameter(data=zp_x, requires_grad=False))
            layer.register_parameter('zps_w', torch.nn.Parameter(data=zps_w, requires_grad=False))

            layer.register_parameter('scales_b', torch.nn.Parameter(data=scale_x * scales_w, requires_grad=False))
            layer.register_parameter('scales_w', torch.nn.Parameter(data=scales_w, requires_grad=False))
            if name == 'none':
                layer.register_parameter('shifts', torch.nn.Parameter(data=torch.tensor([0], device='cuda'), requires_grad=False))
                layer.register_parameter('mults', torch.nn.Parameter(data=torch.tensor([1], device='cuda'), requires_grad=False))
                layer.register_parameter('zp_x_next', torch.nn.Parameter(data=torch.tensor(0, device='cuda'), requires_grad=False))
            else:
                layer.register_parameter('shifts', torch.nn.Parameter(data=shifts, requires_grad=False))
                layer.register_parameter('mults', torch.nn.Parameter(data=mults, requires_grad=False))
                layer.register_parameter('zp_x_next', torch.nn.Parameter(data=zp_x_next, requires_grad=False))

            if type(layer) in supported_modules:
                if type(layer) == nn.Conv2d:
                    # The kernels are identical along the layer.in_channels axis, that could be made more efficiently
                    layer.zp_w_kernel = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                                  dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)

                    for out_channel in range(layer.out_channels):
                        layer.zp_w_kernel.weight.data[out_channel].fill_(zps_w[out_channel])

                    layer.zp_w_kernel.weight.data = layer.zp_w_kernel.weight.data.cuda()
                    layer.unbiased_layer = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                                     dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)
                    layer.unbiased_layer.weight.data = layer.weight.data

                if type(layer) == nn.Linear:
                    layer.zp_w_kernel = nn.Linear(layer.in_features, layer.out_features, bias=False)
                    layer.zp_w_kernel.weight.data.fill_(zps_w[0])
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
                if type(qlayer) == nn.Conv2d:
                    model.seq[i].weight.data = dequantize_4d_tensor_per_channel(qlayer.weight.data, qlayer.scales_w, qlayer.zps_w)
                elif type(qlayer) == nn.Linear:
                    model.seq[i].weight.data = dequantize_tensor(qlayer.weight.data, qlayer.scales_w, qlayer.zps_w)
                if model.seq[i].bias is not None:
                    model.seq[i].bias.data = dequantize_tensor(qlayer.bias.data, qlayer.scales_b, torch.tensor([0], device='cuda'))


def update_qmodel(qmodel, model):
    with torch.no_grad():
        for i, layer in enumerate(model.seq):
            if type(layer) in supported_modules:  # Only these layers have been modified
                if type(layer) == nn.Conv2d:
                    qmodel.seq[i].weight.data = quantize_tensor_4d(model.seq[i].weight.data, qmodel.seq[i].scales_w, qmodel.seq[i].zps_w)
                elif type(layer) == nn.Linear:
                    qmodel.seq[i].weight.data = quantize_tensor(model.seq[i].weight.data, qmodel.seq[i].scales_w, qmodel.seq[i].zps_w)
                if qmodel.seq[i].bias is not None:
                    qmodel.seq[i].bias.data = quantize_tensor(model.seq[i].bias.data, qmodel.seq[i].scales_b, 0)
                qmodel.seq[i].unbiased_layer.weight.data = qmodel.seq[i].weight.data
                for p in qmodel.seq[i].unbiased_layer.parameters():
                    p.requires_grad = False


def increase_fib_proportion(qmodel, optimizer, bits, proportions, step, strategy='quantile'):
    with torch.no_grad():
        idx = 0
        for layer in qmodel.seq:
            if type(layer) in supported_modules:
                for name, param in layer.named_parameters():
                    if name == 'weight':
                        param.data, new_Ts = fib_quantize_tensor(param.data, proportions, step, bits=bits, strategy=strategy)
                        # Multiply element-wise the old Ts and the new Ts such that we always keep at 0 the Ts that already were 0
                        optimizer.param_groups[0]['Ts'][idx] = torch.mul(optimizer.param_groups[0]['Ts'][idx], new_Ts)
                        # print(optimizer.param_groups[0]['Ts'][idx].sum() / np.prod(optimizer.param_groups[0]['Ts'][idx].shape))
                    if name == 'weight' or name == 'bias':
                        idx += 1
                layer.unbiased_layer.weight.data = layer.weight.data

            else:
                for _ in layer.parameters():
                    idx += 1


def qlayer_forward(q_x, layer, layer_stats=None, use_mean=False):
    if not type(layer) in supported_modules:
        raise TypeError("qlayer_forward not implemented for layer of type {}".format(type(layer).__name__))

    gathering_stats = layer_stats is not None
    log = False
    if log and not gathering_stats:
        print(Color.YELLOW + "x_min=" + repr(q_x.min().item()) + ", x_max=" + repr(q_x.max().item()) + Color.END)

    part1 = layer(q_x)

    if type(layer) == nn.Linear:  # For linear layers only
        q_x_sum = torch.sum(q_x, dim=1)
        assert len(layer.zps_w == 1)
        part2 = torch.unsqueeze(q_x_sum * layer.zps_w[0], dim=1)
    elif type(layer) == nn.Conv2d:
        # Apply the convolution with the zp_w_kernel
        part2 = layer.zp_w_kernel(q_x)

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
    if type(layer) == nn.Linear:
        output = ((layer.mults[0] * result.long()) >> layer.shifts[0]).float() + layer.zp_x_next
    elif type(layer) == nn.Conv2d:
        # result shape: n x c x h x w
        # layer.mults, layer.shifts, scales shape: c
        scales = torch.mul(layer.mults, 1 / (2 ** layer.shifts))
        # TODO: use int multiplication and shift instead of fp multiplication
        output = torch.einsum("c,nchw->nchw", scales, result.long()) + unsqueeze_1d_to_4d(layer.zp_x_next.unsqueeze(0), dim=1)

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

    scale_x, zp_x = calc_scale_zero_point(qmodel.low_val_input, qmodel.high_val_input, bits=bits, fib=False)
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
            layer.register_parameter('part3', torch.nn.Parameter(layers_means[i]['part3'], requires_grad=False))
            layer.register_parameter('part4', torch.nn.Parameter(layers_means[i]['part4'], requires_grad=False))
            i += 1

    return qmodel  # Works in place but still returns qmodel


# This function returns the proportion of fib weights in a given qmodel, averaged over the layers
# The idea of using weighted=False is that all layers are equally important no matter how many weights they have
# Weighted=True gives the true proportion of fib weights in the qmodel
def average_proportion_fib(qmodel, weighted=False):
    bits = qmodel.bits.item()
    proportions = []
    weights = []
    for layer in qmodel.seq:
        if type(layer) in supported_modules:
            n = np.prod(layer.weight.shape) if weighted else 1
            proportions.append(proportion_fib(layer.weight, bits=bits) * n)
            weights.append(n)

    return sum(proportions) / sum(weights)


# This function returns the average distance to fib weights in a given qmodel, averaged over the layers
# The idea of using weighted=False is that all layers are equally important no matter how many weights they have
# Weighted=True gives the true average distance to fib weights in the qmodel
def average_distance_fib(qmodel, weighted=False):
    bits = qmodel.bits.item()
    avg_distances = []
    weights = []
    for layer in qmodel.seq:
        if type(layer) in supported_modules:
            _, distances = fib_distances(layer.weight, bits)
            avg_fib_dist = distances.mean().item()
            n = np.prod(layer.weight.shape) if weighted else 1
            avg_distances.append(avg_fib_dist * n)
            weights.append(n)

    return sum(avg_distances) / sum(weights)

