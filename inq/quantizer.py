from inq.quantization_util import *
from inq.qforward import *
from examples.mnist_models import *
from examples.supported_modules import supported_modules
from examples.supported_modules import batch_norm_modules
import torch
import copy

ACC_BITS = 32


def compute_qmodel(model, stats, optimizer, dummy_datapoint, device, proportions=None, step=None, bits=8,
                   fib=False, strategy='quantile', scheme='per_out_channel'):
    # Copy the model into qmodel (and its device)
    qmodel = copy.deepcopy(model)

    # Choose which stat to use
    low_key = 'min'
    high_key = 'max'

    layers_to_quantize = []
    stat_names = []
    indices = []  # Contains the id of the parameters that will be quantized (as stored in the optimizer)

    # Compute the min and max value of the quantized activation (0 and 255 for example for 8 bits) and register them as parameters
    # Note that we use fib=False because the activations are not fib encoded
    activation_qmin, activation_qmax = calc_qmin_qmax(device, bits=bits, negative=False, fib=False)
    qmodel.register_parameter('activation_qmin', nn.Parameter(activation_qmin, requires_grad=False))
    qmodel.register_parameter('activation_qmax', nn.Parameter(activation_qmax, requires_grad=False))
    with torch.no_grad():
        handle_batch_norms(qmodel)
        i = 0
        param_idx = 0
        for layer in qmodel.seq:
            if type(layer) in supported_modules:
                for name, _ in layer.named_parameters():
                    if name == 'weight':
                        # We are only interested in quantizing the weight so we only store only weight param index
                        indices.append(param_idx)
                    if name == 'weight' or name == 'bias':
                        # These are the only 2 parameters that are in the param_groups of the optimizer
                        # The other parameters are added during quantization and therefore or not in the optimizer
                        # The indices list is used to access the optimizer's parameters so we only increment idx
                        # if the parameter exists in the optimizer as well.
                        param_idx += 1
                # there is a shift of 1 in the name: for layer conv1 we use stats['conv2'] for example for the original MNIST net.
                if len(layers_to_quantize) > 0:
                    stat_names.append(repr(i))
                layers_to_quantize.append(layer)
                i += 1

            else:
                for _ in layer.parameters():
                    param_idx += 1

        stat_names.append('none')  # Dummy stat name to indicate that we are at the last layer and we do not actually need the stat

        # Initialization
        qmodel.register_parameter('bits', nn.Parameter(data=torch.tensor(bits), requires_grad=False))
        qmodel.register_parameter('low_val_input', nn.Parameter(data=stats['0'][low_key], requires_grad=False))
        qmodel.register_parameter('high_val_input', nn.Parameter(data=stats['0'][high_key], requires_grad=False))
        scale_x, zp_x = calc_scale_zero_point(qmodel.low_val_input, qmodel.high_val_input, device, bits=bits)
        qmodel.register_parameter('input_scale_x', nn.Parameter(data=scale_x, requires_grad=False))
        qmodel.register_parameter('input_zp_x', nn.Parameter(data=zp_x, requires_grad=False))

        assert(len(stat_names) == len(layers_to_quantize) and len(layers_to_quantize) == len(indices))

        for layer_idx, (name, layer, param_idx) in enumerate(zip(stat_names, layers_to_quantize, indices)):
            print_quantization(fib, proportions, step, bits, layer_idx, len(layers_to_quantize), type(layer))
            if name == 'none':
                scale_x_next, zp_x_next = torch.tensor(1.0, device=device), torch.tensor(0, device=device)
            else:
                scale_x_next, zp_x_next = calc_scale_zero_point(stats[name][low_key], stats[name][high_key], device, bits=bits, fib=False)

            q_w, q_b, shifts, mults, zps_w, scales_w, scale_x, combined_scales, Ts = \
                compute_quantized_layer(layer, scale_x, scale_x_next, device,
                                        proportions=proportions, step=step, bits=bits,
                                        fib=fib, strategy=strategy, scheme=scheme)

            optimizer.param_groups[0]['Ts'][param_idx] = Ts
            layer.weight.data = q_w
            if layer.bias is not None:
                layer.bias.data = q_b
            layer.register_parameter('zp_x', torch.nn.Parameter(data=zp_x, requires_grad=False))
            layer.register_parameter('zps_w', torch.nn.Parameter(data=zps_w, requires_grad=False))
            layer.register_parameter('scale_x', torch.nn.Parameter(data=scale_x, requires_grad=False))

            layer.register_parameter('scales_b', torch.nn.Parameter(data=scale_x * scales_w, requires_grad=False))
            layer.register_parameter('scales_w', torch.nn.Parameter(data=scales_w, requires_grad=False))
            layer.register_parameter('scale_x_next', torch.nn.Parameter(data=scale_x_next, requires_grad=False))
            if name == 'none':
                layer.register_parameter('shifts', torch.nn.Parameter(data=torch.tensor([0], device=device), requires_grad=False))
                layer.register_parameter('mults', torch.nn.Parameter(data=torch.tensor([1], device=device), requires_grad=False))
                layer.register_parameter('zp_x_next', torch.nn.Parameter(data=torch.tensor(0, device=device), requires_grad=False))
                layer.register_parameter('is_last', torch.nn.Parameter(data=torch.tensor(True), requires_grad=False))
            else:
                layer.register_parameter('shifts', torch.nn.Parameter(data=shifts, requires_grad=False))
                layer.register_parameter('mults', torch.nn.Parameter(data=mults, requires_grad=False))
                layer.register_parameter('zp_x_next', torch.nn.Parameter(data=zp_x_next, requires_grad=False))
                layer.register_parameter('is_last', torch.nn.Parameter(data=torch.tensor(False), requires_grad=False))

            if type(layer) in supported_modules:
                if type(layer) == nn.Conv2d:
                    # The kernels are identical along the layer.in_channels axis, that could be made more efficiently
                    zp_out_channels = layer.out_channels if scheme == 'per_out_channel' else 1
                    layer.zp_w_kernel = nn.Conv2d(layer.in_channels, zp_out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                                  dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)

                    for out_channel in range(zp_out_channels):
                        layer.zp_w_kernel.weight.data[out_channel].fill_(zps_w[out_channel])

                    layer.zp_w_kernel.weight.data = layer.zp_w_kernel.weight.data.to(device)
                    layer.unbiased_layer = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                                     dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)
                    layer.unbiased_layer.weight.data = layer.weight.data

                if type(layer) == nn.Linear:
                    layer.zp_w_kernel = nn.Linear(layer.in_features, layer.out_features, bias=False)
                    layer.zp_w_kernel.weight.data.fill_(zps_w[0])
                    layer.zp_w_kernel.weight.data = layer.zp_w_kernel.weight.data.to(device)
                    layer.unbiased_layer = nn.Linear(layer.in_features, layer.out_features, bias=False)
                    layer.unbiased_layer.weight.data = layer.weight.data

                for p in layer.zp_w_kernel.parameters():
                    p.requires_grad = False
                for p in layer.unbiased_layer.parameters():
                    p.requires_grad = False

            scale_x = scale_x_next
            zp_x = zp_x_next
    print_quantization(fib, proportions, step, bits, layer_idx, len(layers_to_quantize), type(layer), persistent=True)
    precompute_constants(qmodel, dummy_datapoint)

    return qmodel


def compute_quantized_layer(layer, scale_x, scale_x_next, device, proportions=None, step=None, bits=8,
                            fib=False, strategy='quantile', scheme='per_out_channel'):
    if not type(layer) in supported_modules:
        raise TypeError("compute_quantized_layer not implemented for layer of type {}".format(type(layer).__name__))

    if type(layer) == nn.Linear:
        low_vals_w, high_vals_w = layer.weight.data.min().unsqueeze(0), layer.weight.data.max().unsqueeze(0)
    else:
        if scheme == 'per_out_channel':
            low_vals_w = stats_over_4d_tensor_per_channel(layer.weight.data, torch.min, device)
            high_vals_w = stats_over_4d_tensor_per_channel(layer.weight.data, torch.max, device)
        else:
            low_vals_w = layer.weight.data.min().unsqueeze(0).expand(layer.out_channels)  # Still expand to match the number of out_channels
            high_vals_w = layer.weight.data.max().unsqueeze(0).expand(layer.out_channels)

    scales_w, zps_w = calc_scales_zero_points(low_vals_w, high_vals_w, device, bits=bits, fib=fib)
    if type(layer) == nn.Conv2d:
        q_w = quantize_tensor_4d(layer.weight.data, scales_w, zps_w)
    elif type(layer) == nn.Linear:
        q_w = quantize_tensor(layer.weight.data, scales_w, zps_w)

    if layer.bias is not None:
        q_b = quantize_tensor(layer.bias.data, scales_w * scale_x, torch.tensor([0], device=device))
    else:
        q_b = None

    if layer.has_bn:
        bn_mults = layer.bn_mults
    else:
        bn_mults = torch.tensor(1., device=device)

    combined_scales = scale_x * scales_w * bn_mults / scale_x_next
    best_mults, best_shifts = get_mults_shifts(combined_scales, device, mult_bits=bits, shift_bits=ACC_BITS)
    # Fibonacci encode the weights (this is very under efficient due to apply_ not working on cuda)
    if fib:
        q_w, Ts = fib_encode_tensor(q_w, proportions, step, device, bits=bits, strategy=strategy)
    else:
        Ts = torch.ones_like(q_w)

    return q_w, q_b, best_shifts, best_mults, zps_w, scales_w, scale_x, combined_scales, Ts


def handle_batch_norms(qmodel):
    for i in range(len(qmodel.seq)):
        if type(qmodel.seq[i]) in supported_modules:
            j = 1

            found_bn = False
            while (i + j) < len(qmodel.seq) and not type(qmodel.seq[i + j]) in supported_modules:
                if type(qmodel.seq[i + j]) in batch_norm_modules:
                    if found_bn:
                        print("ERROR: multiple batch norm layers found for one quantized layer")
                    found_bn = True
                    # qmodel.seq[i].bn = qmodel.seq[i + j]  # Attach the batch norm layer to the conv2d layer
                    qmodel.seq[i].register_parameter('bn_mults', torch.nn.Parameter(
                        qmodel.seq[i + j].weight / ((qmodel.seq[i + j].running_var ** 0.5) - qmodel.seq[i + j].eps), requires_grad=False))
                    qmodel.seq[i].register_parameter('bn_add', torch.nn.Parameter(
                        qmodel.seq[i + j].bias / qmodel.seq[i].bn_mults - qmodel.seq[i + j].running_mean, requires_grad=False))
                    # qmodel.seq[i].bn_bias = qmodel.seq[i + j].bias
                    # qmodel.seq[i].bn_rm = qmodel.seq[i + j].running_mean
                    # qmodel.seq[i + j] = nn.Identity()  # Remove the batch norm layer from the sequential model (replace it by identity)
                j += 1
            qmodel.seq[i].register_parameter('has_bn', nn.Parameter(data=torch.tensor(found_bn), requires_grad=False))


def precompute_constants(qmodel, dummy_datapoint):
    qmodel.eval()
    with torch.no_grad():
        constants = qmodel_forward(qmodel, dummy_datapoint, computing_constants=True)

    i = 0
    for layer in qmodel.seq:  # Only iterate over the main modules and not the modules contained in those
        if type(layer) in supported_modules:
            layer.register_parameter('constant', torch.nn.Parameter(constants[i], requires_grad=False))
            i += 1


def increase_fib_proportion(qmodel, optimizer, bits, proportions, step, device, strategy='quantile'):
    with torch.no_grad():
        idx = 0
        for layer in qmodel.seq:
            if type(layer) in supported_modules:
                for name, param in layer.named_parameters():
                    if name == 'weight':
                        param.data, new_Ts = fib_encode_tensor(param.data, proportions, step, device, bits=bits, strategy=strategy)
                        # Multiply element-wise the old Ts and the new Ts such that we always keep at 0 the Ts that already were 0
                        optimizer.param_groups[0]['Ts'][idx] = torch.mul(optimizer.param_groups[0]['Ts'][idx], new_Ts)
                        # print(optimizer.param_groups[0]['Ts'][idx].sum() / np.prod(optimizer.param_groups[0]['Ts'][idx].shape))
                    if name == 'weight' or name == 'bias':
                        idx += 1
                layer.unbiased_layer.weight.data = layer.weight.data

            else:
                for _ in layer.parameters():
                    idx += 1


