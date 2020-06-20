from inq.quantizer import *


# This function descales an int (+fib) quantized network and puts it in its original form
# It also uses the original model before int (+fib) quantization to find the exact expected structure
def update_model(model, qmodel, device):
    with torch.no_grad():
        for i, qlayer in enumerate(qmodel.seq):
            if type(qlayer) in supported_modules:  # Only these layers have been modified
                if type(qlayer) == nn.Conv2d:
                    model.seq[i].weight.data = dequantize_4d_tensor_per_channel(qlayer.weight.data, qlayer.scales_w, qlayer.zps_w)
                elif type(qlayer) == nn.Linear:
                    model.seq[i].weight.data = dequantize_tensor(qlayer.weight.data, qlayer.scales_w, qlayer.zps_w)
                if model.seq[i].bias is not None:
                    model.seq[i].bias.data = dequantize_tensor(qlayer.bias.data, qlayer.scales_b, torch.tensor([0], device=device))


# This function updates the qmodel with the retrained weights from the unscaled model. It does not change
# the quantization parameters (scales and zero_points) so it just has to quantize the unscaled model with
# the original scales and zero_points
def update_qmodel(qmodel, model, device):
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
            elif type(layer) in batch_norm_modules:
                # Update the batch norm layers
                qmodel.seq[i] = copy.deepcopy(model.seq[i])

        # Recompute the bn_mults and bn_add
        handle_batch_norms(qmodel)

        # Recompute combined_scales because bn_mults has changed
        for layer in qmodel.seq:
            if type(layer) in supported_modules:
                if layer.has_bn:
                    bn_mults = layer.bn_mults
                else:
                    bn_mults = torch.tensor(1., device=device)
                combined_scales = layer.scale_x * layer.scales_w * bn_mults / layer.scale_x_next
                mults, shifts = get_mults_shifts(combined_scales, device, mult_bits=qmodel.bits, shift_bits=ACC_BITS)
                if layer.is_last:
                    layer.register_parameter('shifts', torch.nn.Parameter(data=torch.tensor([0], device=device), requires_grad=False))
                    layer.register_parameter('mults', torch.nn.Parameter(data=torch.tensor([1], device=device), requires_grad=False))
                else:
                    layer.register_parameter('shifts', torch.nn.Parameter(data=shifts, requires_grad=False))
                    layer.register_parameter('mults', torch.nn.Parameter(data=mults, requires_grad=False))

