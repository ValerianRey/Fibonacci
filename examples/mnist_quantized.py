import os
import random
import shutil
import time
import warnings
from types import SimpleNamespace

import examples.mnist_models as mnist_models
from examples.print_util import Color, print_train, print_test, print_header
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from inq.stats import *
from inq.quantization import *
from examples.metrics import *

import inq

settings_dict = {
    'workers': 8,  # Increasing that seems to require A LOT of RAM memory (default was 8)
    'epochs': 16,
    'start_epoch': 0,  # Used for faster restart
    'batch_size': 64,  # default was 256
    'val_batch_size': 4096,  # Keep that low to have enough GPU memory for scaling validation
    'stats_batch_size': 1000,  # This should be a divider of the dataset size
    'lr': 0.1,  # Learning rate, default was 0.001
    'gamma': 0.7,  # Multiplicative reduction of the learning rate at each epoch, default was 0.7
    'momentum': 0.1,  # Gradient momentum, default was 0.9
    'weight_decay': 0.0,  # L2 regularization parameter, default was 0.0005
    'print_interval': 1,
    'resume': 'saves/mnist_99_no_data_parallel.pth',  # default was ''
    'seed': None,  # default: None
    'quantize': False,
    'weight_bits': 8,
    'iterative_steps': [0.0, 0.5, 0.75, 0.875, 1],  # at the last step we still need to retrain parameters that are not quantized (like the biases)
    'log_dir': "logs/",
    'tensorboard': False,
    'print_weights_after_quantization': 'no',  # long, short, no
    'print_weights_after_retraining': 'no',  # long, short, no
    'print_weights_end': 'no',  # long, short, no
    'load_stats': True,  # Be very careful to recompute the stats when the quantization scheme changes
    'load_layers_means': False  # Same here
}

best_acc1 = 0


def main():
    args = SimpleNamespace(**settings_dict)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        shuffle = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        shuffle = True

    main_worker(args, shuffle=shuffle)


def main_worker(args, shuffle=True):
    global best_acc1

    model = mnist_models.Net().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = args.epochs
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print(Color.RED + "No checkpoint found at '{}'".format(args.resume) + Color.END)

    cudnn.benchmark = True

    if args.quantize:
        quantized_parameters = []
        full_precision_parameters = []
        for name, param in model.named_parameters():
            if 'bn' in name or 'bias' in name:
                full_precision_parameters.append(param)
            else:
                quantized_parameters.append(param)
        optimizer = inq.SGD([
            {'params': quantized_parameters},
            {'params': full_precision_parameters, 'weight_bits': None}
        ], args.lr, momentum=args.momentum, weight_decay=args.weight_decay, weight_bits=args.weight_bits)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    train_dataset = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    val_dataset = datasets.MNIST('', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle,
                                               num_workers=args.workers, pin_memory=True)

    train_stats_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.stats_batch_size, shuffle=shuffle,
                                                     num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=shuffle,
                                             num_workers=args.workers, pin_memory=True)

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

        # model.print(how=args.print_weights_after_quantization)

        if args.start_epoch < args.epochs:
            print_header(color=Color.GREEN)

        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args)
            scheduler.step()
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            best_acc1 = max(acc1, best_acc1)
        print('\n\n')
        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False, filename='saves/mnist_99_no_data_parallel.pth')

        model.print(how=args.print_weights_after_retraining)

    stats = load_or_gather_stats(model, train_stats_loader, args.load_stats)

    validate(val_loader, model, criterion, args, scale=False)
    validate(val_loader, model, criterion, args, scale=True, stats=stats, fibonacci_encode=False,
             train_stats_loader=train_stats_loader)
    validate(val_loader, model, criterion, args, scale=True, stats=stats, fibonacci_encode=True,
             train_stats_loader=train_stats_loader)

    # Save the graph to Tensorboard
    if args.tensorboard:
        writer = SummaryWriter(args.log_dir)
        images, labels = next(iter(train_loader))
        grid = make_grid(images)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, images.cuda())
        writer.close()

    # Print all the parameters of the neural network to get an idea of how the weights are quantized
    model.print(how=args.print_weights_end)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))

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


def validate(val_loader, model, criterion, args, scale=False, stats=None, fibonacci_encode=False, train_stats_loader=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    color = Color.CYAN if scale and not fibonacci_encode else Color.GREEN if scale and fibonacci_encode else Color.PURPLE
    title = 'Test Int' if scale and not fibonacci_encode else 'Test Fib' if scale and fibonacci_encode else 'Test'

    # switch to evaluate mode
    model.eval()

    if scale:
        qmodel = compute_qmodel(model, stats, num_bits=args.weight_bits, fib=fibonacci_encode)
        qmodel.eval()

        layers_means = load_or_gather_layers_means(qmodel, args, train_stats_loader, load=args.load_layers_means, fibonacci_encode=fibonacci_encode)

        qmodel = enhance_qmodel(qmodel, layers_means)
        qmodel.print(how=args.print_weights_after_quantization)

    with torch.no_grad():
        end = time.time()

        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            if scale:
                data = data.cuda()
                output = qmodel_forward(qmodel, data, num_bits=args.weight_bits)
            else:
                output = model(data)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))

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


if __name__ == '__main__':
    main()
