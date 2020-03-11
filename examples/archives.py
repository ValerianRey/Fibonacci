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
