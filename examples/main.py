import os
import random
from types import SimpleNamespace

import examples.mnist_models as mnist_models
import examples.cifar10_models as cifar10_models
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
from inq.quantizer import *
from examples.metrics import *
from examples.dpn import *
from examples.preact_resnet import *
from os import path
from inq.retraining import *

import inq

settings_dict = {
    'dataset': 'cifar10',  # 'mnist', 'cifar10'
    'arch': 'PARN18_nores',  # 'Net', 'Net_sigmoid', 'Net_tanh', 'LeNet', 'LeNetDropout', 'DPN26' (very slow), 'DPN92' (giga slow), 'PARN18', 'PARN18_nores'
    'workers': 4,  # Increasing that seems to require A LOT of RAM memory (default was 8)
    'epochs': 100,
    'retrain_epochs': 10,
    'start_epoch': 0,  # Used for faster restart
    'batch_size': 64,  # default was 256
    'val_batch_size': 256,  # Keep that low to have enough GPU memory for scaling validation
    'stats_batch_size': 1000,  # This should be a divider of the dataset size
    'lr': 0.1,  # Learning rate, default was 0.001
    'lr_retrain': 0.05,
    'gamma': 0.92,  # Multiplicative reduction of the learning rate at each epoch, default was 0.7, 0.95 for cifar10 is good
    'gamma_retrain': 0.75,
    'momentum': 0.9,  # Gradient momentum, default was 0.9
    'momentum_retrain': 0.5,
    'weight_decay': 0.0005,  # L2 regularization parameter, default was 0.0005
    'weight_decay_retrain': 0.0005,
    'print_interval': 1,
    'val_interval': 5,  # Use a large value if you want to avoid wasting time computing the test accuracy and printing it.
    'seed': None,  # default: None
    'quantize': True,
    'strategy': 'reverse_quantile',
    'weight_bits': 8,
    'iterative_steps': [1.],  #[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0],  # [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.993, 0.995, 0.998, 0.999, 1.0],  # np.arange(0.4, 1.0, 0.03).tolist() + [0.994, 0.997, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 1.0],  # [0.33, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 1.0],
    'log_dir': "logs/",
    'tensorboard': False,
    'pretrain': False,
    'load_model': True,
    'load_stats': True,  # Be very careful to recompute the stats when the quantization scheme changes
    'load_qmodel_fib': False
}


def main():
    args = SimpleNamespace(**settings_dict)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        shuffle = False
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')
    else:
        shuffle = True

    main_worker(args, shuffle=shuffle)


def main_worker(args, shuffle=True):
    global best_acc1

    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST('', train=False, transform=transform)
        if args.arch == 'Net':
            model = mnist_models.Net(non_linearity=nn.ReLU).cuda()
        elif args.arch == 'Net_sigmoid':
            model = mnist_models.Net(non_linearity=nn.Sigmoid).cuda()
        elif args.arch == 'Net_tanh':
            model = mnist_models.Net(non_linearity=nn.Tanh).cuda()
        elif args.arch == 'Net_2':
            model = mnist_models.Net(non_linearity=nn.ReLU).cuda()

    elif args.dataset == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train_dataset = datasets.CIFAR10('CIFAR10', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10('CIFAR10', train=False, download=True, transform=transform)
        if args.arch == 'LeNet' or args.arch == 'LeNet2':
            model = cifar10_models.LeNet(dropout=False).cuda()
        if args.arch == 'LeNetDropout':
            model = cifar10_models.LeNet(dropout=True).cuda()
        if args.arch == 'DPN92':
            model = DPN92().cuda()
        if args.arch == 'DPN26':
            model = DPN26().cuda()
        if args.arch == 'PARN18':
            model = PreActResNet18().cuda()
        if args.arch == 'PARN18_nores':
            model = cifar10_models.parn(depth=18).cuda()
        if args.arch == 'PARN18_nores_noaffine':
            model = cifar10_models.parn(depth=18, affine_batch_norm=False).cuda()

    else:
        raise ValueError("Dataset {} not supported. Use mnist or cifar10.".format(args.dataset))

    saves_path = 'saves/' + args.dataset + '/' + args.arch + '/'
    model_path = saves_path + 'model.pth'
    qmodel_fib_path = saves_path + 'qmodel_fib.pth'

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = inq.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, weight_bits=args.weight_bits)

    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle,
                                               num_workers=args.workers, pin_memory=True)

    train_stats_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.stats_batch_size, shuffle=shuffle,
                                                     num_workers=args.workers, pin_memory=True)
    dummy_datapoint, _ = train_dataset[0]
    dummy_datapoint = dummy_datapoint.unsqueeze(0).cuda()

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=shuffle,
                                             num_workers=args.workers, pin_memory=True)

    print_dataset_name(args.dataset)
    # optionally resume from a checkpoint
    if args.load_model:
        if os.path.isfile(model_path):
            print("Loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}'"
                  .format(model_path))
        else:
            print(Color.RED + "No checkpoint found at '{}'".format(model_path) + Color.END)
    if args.pretrain:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=args.gamma)
        print_header(color=Color.UNDERLINE)
        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args)
            scheduler.step()
            # evaluate on validation set
            if epoch % args.val_interval == 0:
                validate(val_loader, model, criterion, args, title='Test unscaled')

        # Create the directory if it does not exist yet, and then save the learned model
        if not path.exists(saves_path):
            os.makedirs(saves_path)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=model_path)

    stats = load_or_gather_stats(model, train_stats_loader, args.load_stats, saves_path)

    if args.load_qmodel_fib:
        if os.path.isfile(qmodel_fib_path):
            # Here we compute the qmodel from
            print("Computing qmodel from model to be able to copy the save into it")  # TODO: need to improve on that by defining a constructor for empty qmodel
            qmodel_fib = compute_qmodel(model, stats, optimizer, dummy_datapoint, proportions=args.iterative_steps, step=0, bits=args.weight_bits, fib=True, strategy=args.strategy)
            print("Loading checkpoint '{}'".format(qmodel_fib_path))
            checkpoint = torch.load(qmodel_fib_path)
            qmodel_fib.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}'"
                  .format(qmodel_fib_path))
            print_header(Color.UNDERLINE)
            validate(val_loader, qmodel_fib, criterion, args, quantized=True, fib=True, title='Test saved qmodel_fib')
        else:
            print(Color.RED + "No checkpoint found at '{}'".format(qmodel_fib_path) + Color.END)
    else:
        optimizer = inq.SGD(model.parameters(), args.lr_retrain, momentum=args.momentum_retrain, weight_decay=args.weight_decay_retrain, weight_bits=args.weight_bits)
        quantization_epochs = len(args.iterative_steps)



        for qepoch in range(quantization_epochs):
            print()
            print_quantization_epoch(qepoch, quantization_epochs, args.iterative_steps[qepoch] * 100)
            print_header(color=Color.UNDERLINE)
            if qepoch == 0:  # The int quantized qmodel is only produced once
                validate(val_loader, model, criterion, args, title='Test original network')
                qmodel_int = compute_qmodel(model, stats, optimizer, dummy_datapoint, bits=args.weight_bits, fib=False)
                validate(val_loader, qmodel_int, criterion, args, quantized=True, fib=False, title='Test int')
                qmodel_fib = compute_qmodel(model, stats, optimizer, dummy_datapoint, proportions=args.iterative_steps, step=0, bits=args.weight_bits, fib=True, strategy=args.strategy)
            else:
                increase_fib_proportion(qmodel_fib, optimizer, args.weight_bits, args.iterative_steps, qepoch, strategy=args.strategy)

            precompute_constants(qmodel_fib, dummy_datapoint)
            title = 'Test ' + '{0:.5f}'.format(args.iterative_steps[qepoch] * 100).rstrip('0').rstrip('.') + '% fib'
            validate(val_loader, qmodel_fib, criterion, args, quantized=True, fib=True, title=title)

            # Plug the fib quantized values inside of the original model
            update_model(model, qmodel_fib)
            optimizer.reset_lr(args.lr_retrain)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma_retrain)
            optimizer.reset_momentum()
            for epoch in range(args.start_epoch, args.retrain_epochs):
                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch, args, retrain=True)
                scheduler.step()
                # evaluate on validation set
                if epoch % args.val_interval == 0:
                    validate(val_loader, model, criterion, args, title='Test unscaled')

            update_qmodel(qmodel_fib, model)
            precompute_constants(qmodel_fib, dummy_datapoint)
            title = title + ' retrained'
            validate(val_loader, qmodel_fib, criterion, args, quantized=True, fib=True, title=title)
            # print_fib_info(average_proportion_fib(qmodel_fib, weighted=True),
            #                average_proportion_fib(qmodel_fib, weighted=False),
            #                average_distance_fib(qmodel_fib, weighted=True),
            #                average_distance_fib(qmodel_fib, weighted=False))
            # print_seq_model(qmodel_fib, how='short')

        save_checkpoint({
            'state_dict': qmodel_fib.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=qmodel_fib_path)

    # Save the graph to Tensorboard
    if args.tensorboard:
        writer = SummaryWriter(args.log_dir)
        images, labels = next(iter(train_loader))
        grid = make_grid(images)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, images.cuda())
        writer.close()

    # Print all the parameters of the neural network to get an idea of how the weights are quantized
    print_seq_model(qmodel_fib, how='short')


def train(train_loader, model, criterion, optimizer, epoch, args, retrain=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()

    lr = optimizer.param_groups[0]['lr']

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
            print_train(epoch, args.retrain_epochs if retrain else args.epochs, batch_idx, len(train_loader),
                        batch_time, losses, top1, lr, persistent=False)

    print_train(epoch, args.retrain_epochs if retrain else args.epochs, len(train_loader)-1, len(train_loader), batch_time, losses, top1, lr)


def validate(val_loader, model, criterion, args, quantized=False, fib=False, title=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    color = Color.CYAN if quantized and not fib else Color.GREEN if quantized and fib else Color.PURPLE
    if title is None:
        title = 'Test Int' if quantized and not fib else 'Test Fib' if quantized and fib else 'Test'

    model.eval()

    with torch.no_grad():
        end = time.time()

        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            if quantized:
                data = data.cuda()
                output = qmodel_forward(model, data)
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


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


if __name__ == '__main__':
    main()
