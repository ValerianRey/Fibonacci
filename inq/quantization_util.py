from inq.fib_util import *
import warnings


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
        scale = torch.tensor(1.)
    else:
        scale = (high_val - low_val) / (qmax - qmin)

    zp = torch.round((qmin - low_val / scale)).to(torch.int)
    # The need to clamp zp depends on how we handle it. For example in a linear layer zp_w is only multiplied once, so it depends on what
    # piece of hardware we use to make that multiplication (is it important to have it on uint8, or is int32 ok, or even float32?)
    # For a conv layer, more multiplications of zp_w are required, so it might be interesting in the future to fibonacci encode the zp_w_kernel
    # and in that case it would probably be better to have zp_w already clamped.
    # zp_x never needs to be clamped because anyway the zp_x value will be added to a int32 accumulator, which will be clamped before needing to
    # be in uint8 format.
    if zp < qmin:
        warnings.warn('zp less than qmin')
        zp = qmin
    elif zp > qmax:
        warnings.warn('zp more than qmax')
        zp = qmax
    return scale, zp  # zero_point needs to be int


def calc_scales_zero_points(low_vals, high_vals, bits=8, fib=False):
    scales = torch.ones_like(low_vals)
    zps = torch.zeros_like(low_vals, dtype=torch.int)

    for i, (low_val, high_val) in enumerate(zip(low_vals, high_vals)):
        scale, zp = calc_scale_zero_point(low_val, high_val, bits=bits, fib=fib)
        scales[i] = scale
        zps[i] = zp

    return scales, zps


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
    best_mults = torch.ones_like(vals, dtype=torch.int)
    best_shifts = torch.zeros_like(vals, dtype=torch.int)

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
