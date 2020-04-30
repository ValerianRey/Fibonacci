"""Tensorboard save (images + graph)"""
# writer = SummaryWriter(args.log_dir)
# images, labels = next(iter(train_loader))
#
# grid = make_grid(images)
# writer.add_image('images', grid, 0)
# writer.add_graph(model, images)
# writer.close()

"""Original way of constructing the data loaders"""
# if args.distributed:
#     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
# else:
#     train_sampler = None
#
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#     num_workers=args.workers, pin_memory=True, sampler=train_sampler)
#
# val_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(valdir, transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ])),
#     batch_size=args.batch_size, shuffle=False,
#     num_workers=args.workers, pin_memory=True)

"""Manual dataset splitting and dataloader construction"""
# from torch.utils.data.sampler import SubsetRandomSampler
# train_dataset = datasets.MNIST('', download=True)
# val_dataset = datasets.MNIST('', download=True)
# # PROBLEM: val_dataset and train_dataset should be transformed into Tensors instead of images
#
# num_train = len(train_dataset)
# indices = list(range(num_train))
# split = int(np.floor(args.valid_proportion * num_train))
#
# if args.random_valid:
#     np.random.seed(args.seed)
#     np.random.shuffle(indices)
#
# train_idx, valid_idx = indices[split:], indices[:split]
#
# train_sampler = SubsetRandomSampler(train_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)
#
# train_loader = torch.utils.data.DataLoader(train_dataset,
#                                            batch_size=args.batch_size, sampler=train_sampler,
#                                            num_workers=args.workers, pin_memory=True)
#
# val_loader = torch.utils.data.DataLoader(val_dataset,
#                                            batch_size=args.batch_size, sampler=valid_sampler,
#                                            num_workers=args.workers, pin_memory=True)

"""Original dataset construction"""
# Data loading code
# print("ARGS DATA = " + repr(args.data))
# traindir = os.path.join(args.data, 'train')
# traindir = args.data + "/train"
# valdir = os.path.join(args.data, 'val')
# valdir = args.data + "/val"
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#
# train_dataset = datasets.ImageFolder(
#     traindir,
#     transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]))

"""Original model creation"""
# create model
# if args.pretrained:
#     print("=> using pre-trained model '{}'".format(args.arch))
#     model = models.__dict__[args.arch](pretrained=True)
# else:
#     print("=> creating model '{}'".format(args.arch))
#     model = models.__dict__[args.arch]()

"""Original weight printing"""
# conv1 = model.module.conv1
# weights = conv1.weight
# print(conv1)
# print(weights)

"""Original scaling quantization inside of quantize_layer"""
# x = scale_x * (x.float() - zp_x)  # Dequantize x
# layer.weight.data = scale_w * (layer.weight.data - zp_w)  # Dequantize the layer weights
# layer.bias.data = scale_b * (layer.bias.data - zp_b)  # Dequantize the layer biases

# All int computation ???
# x = (layer(x) / scale_next) + zero_point_next  # Forward pass the layer and quantize the result

"""Scale tensor"""
# def scale_tensor(x, bits=8):
# qmax = 2 ** bits - 1
# qmin = 0
# with torch.no_grad():
#     min_val = torch.min(x).item()
#     max_val = torch.max(x).item()
#     scale = (max_val - min_val) / (qmax - qmin)
#     zero_point = int(max(min(qmin - min_val / scale, qmax), qmin))
#     q_x = zero_point + x / scale
#     q_x.clamp_(qmin, qmax).round_()
#     q_x = q_x.round()
#
# return q_x

"""Negative number proportion in tensor"""
# neg_occurences = sum([1 if (value < 0) else 0 for value in layer.weight.data.view(-1)])
# print('Negative number percentage: ' + repr(neg_occurences/len(layer.weight.data.view(-1))))

"""Original int quantization scheme"""
# def quantize_layer(x, layer, stat, scale_x, zp_x, num_bits=8, fibonacci_encode=False):
#     # for both conv and linear layers
#
#     # cache old values
#     W = layer.weight.data
#     B = layer.bias.data
#
#     # # quantise weights, activations are already quantised
#     # w, scale_w, zp_w = quantize_tensor(layer.weight.data, num_bits=num_bits)
#     # b, scale_b, zp_b = quantize_tensor(layer.bias.data, num_bits=num_bits)
#     #
#     # # Turn the layer and activation into float type (even though the numbers are actually integers)
#     # layer.weight.data = w.float()
#     # layer.bias.data = b.float()
#     # x = x.float()
#     #
#     # # Compute scale and zero_point from min and max statistics
#     # scale_next, zero_point_next = calc_scale_zero_point(min_val=stat['min'], max_val=stat['max'], num_bits=num_bits)
#     # combined_scale = scale_x.item() * scale_w.item() / scale_next.item()
#     # best_mult, best_shift = get_mult_shift(combined_scale, num_bits, num_bits)
#     # layer.weight.data = best_mult * (layer.weight.data - zp_w)
#     # layer.bias.data = best_mult * (layer.bias.data - zp_b)
#     #
#     # # Fibonacci encode the weights (this is very under efficient due to apply_ not working on cuda)
#     # if fibonacci_encode:
#     #     layer.weight.data = layer.weight.data.char().cpu()
#     #     layer.weight.data.apply_(fib_code_int)
#     #     layer.weight.data = layer.weight.data.float().cuda()
#
#     layer.weight.data, layer.bias.data, best_shift, zero_point_next, scale_next\
#         = compute_quantized_layer(layer, stat, scale_x, num_bits=num_bits, fibonacci_encode=fibonacci_encode)
#
#     x = x.float()
#     x = x - zp_x
#     # All int computation
#     x = ((layer(x) >> best_shift).round().int() + zero_point_next).float()
#
#     # Reset weights for next forward pass
#     layer.weight.data = W
#     layer.bias.data = B
#
#     return x, scale_next, zero_point_next
#
#
# def quant_forward(model, x, stats, num_bits=8, fibonacci_encode=False):
#     # Quantise before inputting into incoming layers
#     x, scale, zero_point = quantize_tensor(x, num_bits=num_bits, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])
#
#     x, scale_next, zero_point_next = quantize_layer(x, model.conv1, stats['conv2'], scale, zero_point, num_bits=num_bits, fibonacci_encode=fibonacci_encode)
#     x = F.relu(x)
#
#     x, scale_next, zero_point_next = quantize_layer(x, model.conv2, stats['fc1'], scale_next, zero_point_next, num_bits=num_bits, fibonacci_encode=fibonacci_encode)
#     x = F.max_pool2d(x, 2)
#     x = model.dropout1(x)
#     x = torch.flatten(x, 1)
#
#     x, scale_next, zero_point_next = quantize_layer(x, model.fc1, stats['fc2'], scale_next, zero_point_next, num_bits=num_bits, fibonacci_encode=fibonacci_encode)
#     x = F.relu(x)
#     x = model.dropout2(x)
#
#     # Back to dequant for final layer
#     x = dequantize_tensor(x, scale_next, zero_point_next)
#
#     x = model.fc2(x)
#
#     return F.log_softmax(x, dim=1)

"""Original qlayer forward (messy)"""
# x = x - layer.zp
# All int computation
# x = (((layer.mult * layer(x).int()) / (2 ** layer.shift)) + layer.zp_next).float()
# x = ((layer.mult * layer(x).int()) // (2 ** layer.shift)).float()  # TODO: verify that this is not wrong (compared to line above + keeping zp_next)

# l_x = layer(x).int() + (layer.zp_w * x.sum(axis=3)).int()
# x = ((layer.mult * l_x) // (2 ** layer.shift)).float()  # TODO: verify that this is not wrong (compared to line above + keeping zp_next)

# l_x = layer(x) - layer.zp_w_kernel(x)


"""Quantization.py when use_mean was optionnal"""
# from inq.fib_util import *
# from examples.mnist_models import *
#
# ACC_BITS = 32
#
#
# def calc_qmin_qmax(num_bits=8, negative=False, fib=False):
#     if negative:
#         qmin = - 2 ** (num_bits - 1)
#         qmax = 2. ** (num_bits - 1) - 1
#     else:
#         qmin = 0
#         qmax = 2 ** num_bits - 1
#
#     if fib:
#         qmin = 0
#         qmax = (qmax + fib_code_int_down(qmax)) // 2  # We do that to not induce a bias by the choice of qmax
#     return qmin, qmax
#
#
# def calc_scale_zero_point(low_val, high_val, num_bits=8, fib=False):
#     qmin, qmax = calc_qmin_qmax(num_bits=num_bits, fib=fib)
#     scale = (high_val - low_val) / (qmax - qmin)
#
#     # TODO: should we clamp the zero_point to [qmin, qmax]?
#     zp = int((qmin - low_val / scale).round())
#     return scale, zp  # zero_point needs to be int
#
#
# # TODO: change that so that it takes as parameters qmin and qmax instead of numbers of bits
# def get_mult_shift(val, num_mult_bits=8, num_shift_bits=32):
#     best_mult = 1
#     best_shift = 0
#     best_diff = abs(val - best_mult)
#     for mult in range(1, 2 ** num_mult_bits):
#         for shift in range(0, num_shift_bits):
#             s_val = val * (2 ** shift)
#             if abs(s_val - mult) < best_diff:
#                 best_diff = abs(s_val - mult)
#                 best_mult = mult
#                 best_shift = shift
#
#     return best_mult, best_shift
#
#
# def quantize_tensor(x, scale, zp):
#     return (zp + (x / scale)).round()
#
#
# def dequantize_tensor(q_x, scale, zp):
#     return scale * (q_x.float() - zp)
#
#
# def compute_quantized_layer(layer, low_val, high_val, scale_x, num_bits=8, fib=False):
#     low_val_w, high_val_w = layer.weight.data.min(), layer.weight.data.max()
#     scale_w, zp_w = calc_scale_zero_point(low_val_w, high_val_w, num_bits=num_bits, fib=fib)
#     q_w = quantize_tensor(layer.weight.data, scale_w, zp_w)
#     q_b = quantize_tensor(layer.bias.data, scale_w * scale_x, 0)
#
#     # Compute scale and zero_point from min and max statistics
#     scale_next, zp_next = calc_scale_zero_point(low_val=low_val, high_val=high_val, num_bits=num_bits, fib=False)
#     combined_scale = scale_x.item() * scale_w.item() / scale_next.item()
#
#     if fib:
#         best_mult, best_shift = get_mult_shift(combined_scale, num_bits-1, ACC_BITS)  # TODO: make that properly
#     else:
#         best_mult, best_shift = get_mult_shift(combined_scale, num_bits, ACC_BITS)
#
#     # Fibonacci encode the weights (this is very under efficient due to apply_ not working on cuda)
#     if fib:
#         q_w = q_w.int().cpu().apply_(fib_code_int).float().cuda()
#
#     return q_w, q_b, best_shift, best_mult, zp_next, scale_next, zp_w, combined_scale
#
#
# def compute_qmodel(model, stats, num_bits=8, fib=False):
#     # Copy the model into qmodel (and its device)
#     device = model.conv1.weight.device
#     qmodel = type(model)()  # get a new instance
#     qmodel.load_state_dict(model.state_dict())  # copy weights and stuff
#     qmodel.to(device)
#
#     # Choose which stat to use
#     low_key = 'min'
#     high_key = 'max'
#
#     # Initialization
#     low_val = stats['conv1'][low_key]
#     high_val = stats['conv1'][high_key]
#     scale, zp = calc_scale_zero_point(low_val=low_val, high_val=high_val, num_bits=num_bits)
#
#     qmodel.low_val_input = low_val
#     qmodel.high_val_input = high_val
#
#     layers = []
#     stat_names = []
#
#     for name, layer in qmodel.named_modules():
#         if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
#             if len(layers) > 0:  # there is a shift of 1 in the name: for layer conv1 we use stats['conv2'] for example for the original MNIST net.
#                 stat_names.append(name)
#             layers.append(layer)
#     # stat_names.append(stat_names[-1])  # we use the stats of the last layer twice TODO: check which one to use
#     stat_names.append('out')
#
#     for name, layer in zip(stat_names, layers):
#         stat = stats[name]
#
#         low_val = stat[low_key]
#         high_val = stat[high_key]
#         q_w, q_b, best_shift, best_mult, zp_next, scale_next, zp_w, combined_scale = compute_quantized_layer(layer, low_val, high_val, scale, num_bits=num_bits, fib=fib)
#
#         layer.weight.data = q_w
#         layer.bias.data = q_b
#         layer.shift = best_shift
#         layer.mult = best_mult
#         layer.zp = zp
#         layer.combined_scale = combined_scale  # Just used for testing, this is included already inside of mult and shift
#         layer.zp_next = zp_next
#         layer.zp_w = zp_w
#
#         if type(layer) == nn.Conv2d:
#             # Use only 1 out channel since anyway all kernels are the same
#             layer.zp_w_kernel = nn.Conv2d(layer.in_channels, 1, layer.kernel_size, stride=layer.stride, padding=layer.padding,
#                                           dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)
#             layer.zp_w_kernel.weight.data.fill_(zp_w)
#             layer.zp_w_kernel.weight.data = layer.zp_w_kernel.weight.data.cuda()  # TODO: clean that
#             layer.unbiased_layer = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding,
#                                              dilation=layer.dilation, groups=layer.groups, bias=False, padding_mode=layer.padding_mode)
#             layer.unbiased_layer.weight.data = layer.weight.data
#             layer.sum_dim = 2  # TODO: make that generic
#
#         if type(layer) == nn.Linear:
#             layer.zp_w_kernel = nn.Linear(layer.in_features, layer.out_features, bias=False)
#             layer.zp_w_kernel.weight.data.fill_(zp_w)
#             layer.zp_w_kernel.weight.data = layer.zp_w_kernel.weight.data.cuda()  # TODO: clean that
#             layer.unbiased_layer = nn.Linear(layer.in_features, layer.out_features, bias=False)
#             layer.unbiased_layer.weight.data = layer.weight.data
#             layer.sum_dim = 1  # TODO: make that generic
#
#         scale = scale_next
#         zp = zp_next
#
#     return qmodel
#
#
# def qlayer_forward(x, layer, layer_stats=None, use_mean=False):
#     log = False
#     if log:
#         print(Color.YELLOW + "x_min=" + repr(x.min().item()) + ", x_max=" + repr(x.max().item()) + Color.END)
#
#     q_x = x
#     zp_x_vec = torch.zeros_like(x).fill_(layer.zp)
#
#     part1 = layer(q_x)
#
#     if layer.sum_dim == 1:  # For linear layers only
#         q_x_sum = torch.sum(q_x, dim=layer.sum_dim)
#         part2 = torch.unsqueeze(q_x_sum * layer.zp_w, dim=1)
#     else:
#         part2 = layer.zp_w_kernel(q_x)  # Apply the convolution with the zp_w_kernel (way less computations than with a conv layer since it only has 1 out channel)
#
#     if use_mean:
#         part3 = layer.part3
#         part4 = layer.part4
#     else:
#         part3 = layer.unbiased_layer(zp_x_vec)
#         part4 = layer.zp_w_kernel(zp_x_vec)
#
#     result = part1 - part2 - part3 + part4
#
#     if layer_stats is not None:
#         layer_stats['part3'].append(torch.unsqueeze(torch.mean(part3, dim=0), dim=0))
#         layer_stats['part4'].append(torch.unsqueeze(torch.mean(part4, dim=0), dim=0))
#
#     if log:
#         print(Color.GRAY + 'result_min=' + repr(result.min().item()) + ', result_max=' + repr(result.max().item()) + Color.END)
#
#     # Rescale the result so that: we get rid of the scaling of this layer, and we scale it properly for the next layer
#     output = ((layer.mult * result.int()) >> layer.shift).float() + layer.zp_next
#     # output = result.int() * layer.combined_scale + layer.zp_next  # just to test, TODO: remove that
#
#     if log:
#         print(Color.GRAY + 'output_min=' + repr(output.min().item()) + ', output_max=' + repr(output.max().item()) + Color.END)
#     return output  # result_scaled_for_next_layer is an int32 number
#
#
# def qmodel_forward(qmodel, x, num_bits=8, layers_stats=None):
#     # Quantise before inputting into incoming layers (no dropout since this is never used for training anyway)
#
#     input_qmin, input_qmax = calc_qmin_qmax(num_bits)
#
#     # TODO: choose which one to use
#     # The first line ensures that all x are quantized with the same scale / zp
#     # The second line uses batch quantization
#     # Accuracies seem to be the same
#     scale_x, zp_x = calc_scale_zero_point(qmodel.low_val_input, qmodel.high_val_input, num_bits=num_bits, fib=False)
#     # scale_x, zp_x = calc_scale_zero_point(x.min(), x.max(), num_bits=num_bits, fib=False)
#
#     x = quantize_tensor(x, scale_x, zp_x)
#
#     use_mean = True
#     x = torch.clamp(x, input_qmin, input_qmax)  # Clamp to be sure that we stay within the uint8 range
#     if layers_stats is not None:
#         x = qlayer_forward(x, qmodel.conv1, layers_stats[0])
#     else:
#         x = qlayer_forward(x, qmodel.conv1, use_mean=use_mean)
#     x = F.relu(x)
#
#     x = torch.clamp(x, input_qmin, input_qmax)  # Clamp to be sure that we stay within the uint8 range
#     if layers_stats is not None:
#         x = qlayer_forward(x, qmodel.conv2, layers_stats[1])
#     else:
#         x = qlayer_forward(x, qmodel.conv2, use_mean=use_mean)
#     x = F.max_pool2d(x, 2)
#     x = torch.flatten(x, 1)
#
#     x = torch.clamp(x, input_qmin, input_qmax)  # Clamp to be sure that we stay within the uint8 range
#     if layers_stats is not None:
#         x = qlayer_forward(x, qmodel.fc1, layers_stats[2])
#     else:
#         x = qlayer_forward(x, qmodel.fc1, use_mean=use_mean)
#     x = F.relu(x)
#
#     x = torch.clamp(x, input_qmin, input_qmax)  # Clamp to be sure that we stay within the uint8 range
#     if layers_stats is not None:
#         x = qlayer_forward(x, qmodel.fc2, layers_stats[3])
#     else:
#         x = qlayer_forward(x, qmodel.fc2, use_mean=use_mean)
#
#     return x
#
#
# # For now this works only on the baseline network (not generic) TODO: make that generic
# def enhance_qmodel(qmodel, layers_means):
#     qmodel.conv1.part3 = layers_means[0]['part3']
#     qmodel.conv1.part4 = layers_means[0]['part4']
#
#     qmodel.conv2.part3 = layers_means[1]['part3']
#     qmodel.conv2.part4 = layers_means[1]['part4']
#
#     qmodel.fc1.part3 = layers_means[2]['part3']
#     qmodel.fc1.part4 = layers_means[2]['part4']
#
#     qmodel.fc2.part3 = layers_means[3]['part3']
#     qmodel.fc2.part4 = layers_means[3]['part4']
#
#     return qmodel  # Works in place but still returns qmodel
