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

