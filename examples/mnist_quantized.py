import os
import random
import shutil
import time
import warnings
import pickle
from types import SimpleNamespace

import examples.mnist_models as mnist_models
from examples.print_util import Color, print_train, print_test, print_header
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from inq.stats import *
from inq.quantization import *

import inq

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

settings_dict = {
    'data': '/project_scratch/data/mnist',  # unused
    'arch': 'resnet18',  # unused
    'workers': 8,  # Increasing that seems to require A LOT of RAM memory (default was 8)
    'epochs': 16,
    'start_epoch': 0,  # Used for faster restart
    'batch_size': 64,  # default was 256
    'val_batch_size': 4096,  # Keep that low to have enough GPU memory for scaling validation
    'lr': 0.1,  # Learning rate, default was 0.001
    'gamma': 0.7,  # Multiplicative reduction of the learning rate at each epoch, default was 0.7
    'momentum': 0.1,  # Gradient momentum, default was 0.9
    'weight_decay': 0.0,  # L2 regularization parameter, default was 0.0005
    'print_interval': 1,
    'resume': 'saves/mnist_99.pth',  # default was ''
    'evaluate': False,
    'pretrained': True,  # unused
    'world_size': -1,
    'rank': -1,
    'dist_url': '',
    'dist_backend': '',
    'seed': None,  # default: None
    'gpu': None,
    'multiprocessing_distributed': False,

    'quantize': False,
    'weight_bits': 8,
    'iterative_steps': [0.0, 0.5, 0.75, 0.875, 1],  # at the last step we still need to retrain parameters that are not quantized (like the biases)
    'log_dir': "logs/",
    'tensorboard': False,
    'print_weights_after_quantization': 'short',  # long, short, no
    'print_weights_after_retraining': 'no',  # long, short, no
    'print_weights_end': 'no'  # long, short, no
}

best_acc1 = 0


def main():
    args = SimpleNamespace(**settings_dict)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    para_model = mnist_models.Net()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            para_model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            para_model = torch.nn.parallel.DistributedDataParallel(para_model, device_ids=[args.gpu])
        else:
            para_model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            para_model = torch.nn.parallel.DistributedDataParallel(para_model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        para_model = para_model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            para_model.features = torch.nn.DataParallel(para_model.features)
            para_model.cuda()
        else:
            para_model = torch.nn.DataParallel(para_model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(para_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = args.epochs
            para_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print(Color.RED + "No checkpoint found at '{}'".format(args.resume) + Color.END)

    cudnn.benchmark = True

    if args.quantize:
        quantized_parameters = []
        full_precision_parameters = []
        for name, param in para_model.named_parameters():
            if 'bn' in name or 'bias' in name:
                full_precision_parameters.append(param)
            else:
                quantized_parameters.append(param)
        optimizer = inq.SGD([
            {'params': quantized_parameters},
            {'params': full_precision_parameters, 'weight_bits': None}
        ], args.lr, momentum=args.momentum, weight_decay=args.weight_decay, weight_bits=args.weight_bits)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    train_dataset = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.val_batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.evaluate:
        validate(val_loader, para_model, criterion, args)
        return

    if args.quantize:
        quantization_epochs = len(args.iterative_steps)
        quantization_scheduler = inq.INQScheduler(optimizer, args.iterative_steps, strategy="pruning")
    else:
        quantization_epochs = 1
        quantization_scheduler = None

    for _ in range(quantization_epochs):
        inq.reset_lr_scheduler(scheduler)

        if args.quantize:
            quantization_scheduler.step()

        # model.module.print(how=args.print_weights_after_quantization)

        if args.start_epoch < args.epochs:
            print_header(color=Color.GREEN)

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # train for one epoch
            train(train_loader, para_model, criterion, optimizer, epoch, args)
            scheduler.step()
            # evaluate on validation set
            acc1 = validate(val_loader, para_model, criterion, args)

            # remember best acc@1 and save checkpoint
            best_acc1 = max(acc1, best_acc1)
        print('\n\n')
        save_checkpoint({
            'state_dict': para_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False, filename='saves/mnist_99_old.pth')

        para_model.module.print(how=args.print_weights_after_retraining)

    # stats = gather_stats(para_model.module, val_loader)
    # with open('stats.pickle', 'wb') as handle:
    #     pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Loading stats from save, be sure to remove this when seed is not fixed")
    with open('stats.pickle', 'rb') as handle:
        stats = pickle.load(handle)

    validate(val_loader, para_model, criterion, args, scale=False)
    validate(val_loader, para_model, criterion, args, scale=True, stats=stats, fibonacci_encode=False)
    validate(val_loader, para_model, criterion, args, scale=True, stats=stats, fibonacci_encode=True)

    # Save the graph to Tensorboard
    if args.tensorboard:
        writer = SummaryWriter(args.log_dir)
        images, labels = next(iter(train_loader))
        grid = make_grid(images)
        writer.add_image('images', grid, 0)
        writer.add_graph(para_model.module, images.cuda())
        writer.close()

    # Print all the parameters of the neural network to get an idea of how the weights are quantized
    para_model.module.print(how=args.print_weights_end)


def train(train_loader, para_model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    para_model.train()

    start = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.gpu is not None:
            data = data.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = para_model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.print_interval == 0:
            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()
            print_train(epoch, args.epochs, batch_idx, len(train_loader),
                        batch_time, losses, top1, persistent=False)

    print_train(epoch, args.epochs, len(train_loader)-1, len(train_loader), batch_time, losses, top1)


def validate(val_loader, para_model, criterion, args, scale=False, stats=None, fibonacci_encode=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    color = Color.CYAN if scale and not fibonacci_encode else Color.GREEN if scale and fibonacci_encode else ''
    title = 'Test Int' if scale and not fibonacci_encode else 'Test Fib' if scale and fibonacci_encode else 'Test'

    # switch to evaluate mode
    para_model.eval()

    if scale:
        qmodel = compute_qmodel(para_model.module, stats, num_bits=args.weight_bits, fibonacci_encode=fibonacci_encode)
        qmodel.eval()
        qmodel.print(how=args.print_weights_after_quantization)

    with torch.no_grad():
        end = time.time()
        for batch_idx, (data, target) in enumerate(val_loader):
            if args.gpu is not None:
                data = data.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if scale:
                data = data.cuda()
                output = qmodel_forward(qmodel, data, stats, num_bits=args.weight_bits)
            else:
                output = para_model(data)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_interval == 0:
                print_test(batch_idx, len(val_loader), batch_time, losses, top1, persistent=False, color=color, title=title)

        print_test(len(val_loader)-1, len(val_loader), batch_time, losses, top1, persistent=True, color=color, title=title)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
