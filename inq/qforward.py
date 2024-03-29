from inq.quantization_util import *
from examples.supported_modules import *
from examples.print_util import *


def qmodel_forward(qmodel, x, computing_constants=False, print_clamped_values=False, verbose=False):
    if computing_constants:
        constants = []
    # Quantise before inputting into incoming layers (no dropout since this is never used for training anyway)
    if print_clamped_values:
        print()

    x = quantize_tensor(x, qmodel.input_scale_x, qmodel.input_zp_x).int()
    too_low_sum = 0
    too_high_sum = 0
    i = 0
    for layer in qmodel.seq:
        if type(layer) in supported_modules:
            too_low, too_high = count_out(x, qmodel.activation_qmin, qmodel.activation_qmax, log=print_clamped_values)
            too_low_sum += too_low
            too_high_sum += too_high
            x = torch.clamp(x, qmodel.activation_qmin, qmodel.activation_qmax)  # Clamp to be sure that we stay within the uint8 range
            if computing_constants:
                x, constant = qlayer_forward(x, layer, computing_constant=True)
                constants.append(constant)
            else:
                x = qlayer_forward(x, layer, verbose=verbose)
            i += 1
        elif not type(layer) in batch_norm_modules:  # The batch norm is already manually handled by bn_add and bn_mult
            x = layer(x.float()).round().int()  # Some layers are not implemented for int type, so we have to cast to float

    if computing_constants:
        return constants
    else:
        return x


def qlayer_forward(q_x, layer, computing_constant=False, verbose=False):
    if not type(layer) in supported_modules:
        raise TypeError("qlayer_forward not implemented for layer of type {}".format(type(layer).__name__))

    if verbose:
        print(Color.YELLOW + "x_min=" + repr(q_x.min().item()) + ", x_max=" + repr(q_x.max().item()) + Color.END)

    part1 = layer(q_x.float()).int()

    if type(layer) == nn.Linear:  # For linear layers only
        q_x_sum = torch.sum(q_x, dim=1, dtype=torch.int)
        assert len(layer.zps_w == 1)
        part2 = torch.unsqueeze(q_x_sum * layer.zps_w[0], dim=1)
    elif type(layer) == nn.Conv2d:
        # Apply the convolution with the zp_w_kernel
        part2 = layer.zp_w_kernel(q_x.float()).int()  # Use conv layer for zp_w computation
        # Or use sum pooling and multiply
        # part2 = torch.einsum("o,nihw->nohw", layer.zps_w, F.avg_pool2d(q_x, layer.kernel_size, stride=1)) * layer.kernel_size[0] * layer.kernel_size[1]

    if computing_constant:
        zp_x_vec = torch.zeros_like(q_x, dtype=torch.float).fill_(layer.zp_x)  # Need to use float to be able to forward this through a layer
        part3 = layer.unbiased_layer(zp_x_vec).int()  # Cast back to int
        part4 = layer.zp_w_kernel(zp_x_vec).int()  # Cast back to int
        if layer.has_bn:
            # part5 = unsqueeze_1d_to_4d((layer.bn_bias / layer.bn_mults - layer.bn_rm) / layer.scales_b, dim=1)
            part5 = torch.round(unsqueeze_1d_to_4d(layer.bn_add / layer.scales_b, dim=1)).int()
        else:
            part5 = torch.zeros_like(part3, dtype=torch.int)
        constant = - part3 + part4 + part5
    else:
        constant = layer.constant
    result = part1 - part2 + constant

    if verbose:
        print(Color.GRAY + 'result_min=' + repr(result.min().item()) + ', result_max=' + repr(result.max().item()) + Color.END)

    # POST-QUANTIZATION

    # Rescale the result so that: we get rid of the scaling of this layer, and we scale it properly for the next layer
    # We could use int instead of long for 8 bits (no risk of overflowing the int32 range)
    if type(layer) == nn.Linear:
        if result.device == torch.device('cpu'):
            # The binary shifting operation overflows when the data is on cpu
            output = ((layer.mults[0] * result) // (2 ** layer.shifts[0])) + layer.zp_x_next
        else:
            # This is the real operation that should be implemented on the specialized hardware: a simple binary shifting
            output = ((layer.mults[0] * result) >> layer.shifts[0]) + layer.zp_x_next
    elif type(layer) == nn.Conv2d:
        # result shape: n x c x h x w
        # layer.mults, layer.shifts, scales shape: c

        # Revert mult and shift to the corresponding combined_scale, then apply fp multiplication (faster on cuda gpu, but uses fp multiplication)
        # scales = torch.mul(layer.mults, 1 / (2 ** layer.shifts))
        # output = torch.einsum("c,nchw->nchw", scales, result).int() + unsqueeze_1d_to_4d(layer.zp_x_next.unsqueeze(0), dim=1)
        # Make the computations by only using int product and shifting (faster on specialized hardware if implemented properly, without a for loop)
        multiplied_result = torch.einsum("c,nchw->nchw", layer.mults, result)
        for channel in range(multiplied_result.shape[1]):
            if multiplied_result.device == torch.device('cpu'):
                # The binary shifting operation overflows when the data is on cpu
                multiplied_result[:, channel, :, :] = multiplied_result[:, channel, :, :] // (2 ** layer.shifts[channel])
            else:
                # This is the real operation that should be implemented on the specialized hardware: a simple binary shifting
                multiplied_result[:, channel, :, :] = multiplied_result[:, channel, :, :] >> layer.shifts[channel]
        output = multiplied_result + unsqueeze_1d_to_4d(layer.zp_x_next.unsqueeze(0), dim=1)

    if verbose:
        print(Color.GRAY + 'output_min=' + repr(output.min().item()) + ', output_max=' + repr(output.max().item()) + Color.END)

    if computing_constant:
        return output, constant
    else:
        return output  # output is an int number stored as float


