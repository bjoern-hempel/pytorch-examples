import argparse
import os
import random
import shutil
import time
import warnings
import sys
import pprint

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from datetime import datetime

# some own imports
from __log import *
from __args import *

# pretty printer
pp = pprint.PrettyPrinter(indent=4)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# define some global vars
csv_handler = {}
csv_files = {}
time_start = datetime.datetime.now()
accuracy_best = {'train': 0, 'val': 0, 'train5': 0, 'val5': 0}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--learning-rate-decrease-factor', default=0.1, type=float,
                    help='learning rate decrease factor')
parser.add_argument('--learning-rate-decrease-after', default=30, type=int,
                    help='learning rate decrease after')
parser.add_argument('--linear-layer', default=None, type=int,
                    help='set a linear layer')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--use-cpu', action='store_true',
                    help='Use the cpu instead of available gpu')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--session-name', default='all', type=str,
                    help='set the session name for logging')
parser.add_argument('--csv-path-settings', default=None, type=str,
                    help='path to write the settings csv')
parser.add_argument('--csv-path-summary', default=None, type=str,
                    help='path to write the summary csv')
parser.add_argument('--csv-path-summary-full', default=None, type=str,
                    help='path to write the full summary csv')
parser.add_argument('--csv-path-trained', default=None, type=str,
                    help='path to write the train csv')
parser.add_argument('--csv-path-validated', default=None, type=str,
                    help='path to write the validated csv')
parser.add_argument('--model-path', default=None, type=str,
                    help='the path to save the model')

best_acc1 = 0

settings = {
    'input_size': '224x224',
    'device_name': 'gpu1060'
}


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def main():
    args = analyseArgs(parser.parse_args())

    settings['model_name'] = args.arch

    # write settings to csv file
    if args.csv_path_settings is not None:
        csv_handler['csv_file_settings'] = getCSVHandlerWithHeader(
            args.csv_path_settings,
            {
                'name': 'settings',
                'model_name': settings['model_name'],
                'input_size': settings['input_size'],
                'device_name': settings['device_name']
            },
            False,
            args,
            time_start
        )
    else:
        csv_handler['csv_file_settings'] = None

    # prepare summary csv file
    if args.csv_path_summary is not None:
        csv_handler['csv_file_summary'] = getCSVHandlerWithHeader(
            args.csv_path_summary,
            {
                'name': 'summary',
                'model_name': settings['model_name'],
                'input_size': settings['input_size'],
                'device_name': settings['device_name']
            },
            False,
            args,
            time_start
        )
    else:
        csv_handler['csv_file_summary'] = None

    # prepare summary csv file
    if args.csv_path_summary_full is not None:
        csv_handler['csv_file_summary_full'] = getCSVHandlerWithHeader(
            args.csv_path_summary_full,
            {
                'name': 'summary_full',
                'model_name': settings['model_name'],
                'input_size': settings['input_size'],
                'device_name': settings['device_name']
            },
            False,
            args,
            time_start
        )
    else:
        csv_handler['csv_file_summary_full'] = None

    # prepare validated csv file
    if args.csv_path_validated is not None:
        csv_handler['csv_file_validated'] = getCSVHandlerWithHeader(
            args.csv_path_validated,
            {
                'name': 'validated',
                'model_name': settings['model_name'],
                'input_size': settings['input_size'],
                'device_name': settings['device_name']
            },
            False,
            args,
            time_start
        )
    else:
        csv_handler['csv_file_validated'] = None

    # write settings
    if csv_handler['csv_file_settings'] is not None:
        for prop in getClassProperties(args):
            writeCSV(csv_handler['csv_file_settings'], False, prop, args.__dict__[prop])

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
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    input_size = 224

    # set linear layer for output
    if args.linear_layer is not None:

        if args.arch in ['resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50']:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, args.linear_layer)
            input_size = 224

        elif args.arch in ['inception_v3']:
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, args.linear_layer)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, args.linear_layer)
            input_size = 299

        elif args.arch in ['alexnet']:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, args.linear_layer)
            input_size = 224

        elif args.arch in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, args.linear_layer)
            input_size = 224

        elif args.arch in ['squeezenet1_0', 'squeezenet1_1']:
            model.classifier[1] = nn.Conv2d(512, args.linear_layer, kernel_size=(1, 1), stride=(1, 1))
            model.num_classes = args.linear_layer
            input_size = 224

        elif args.arch in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, args.linear_layer)
            input_size = 224

        else:
            print('Invalid model name "{}", exiting...'.format(args.arch))
            exit()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model)

            if not args.use_cpu:
                model = model.cuda()


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    if not args.use_cpu:
        criterion = criterion.cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(valdir, transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, checkpoint['epoch'])
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    correct1_all = 0
    time_all = 0
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        if not args.use_cpu:
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        correct1, correct5 = accuracy_count(output, target, topk=(1, 5))
        correct1_all += correct1
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        time_all += batch_time.val

        # print output
        if i % args.print_freq == 0:
            print('Train ({top1.avg:.3f}%/{0:.3f}%): [{1}/{2}][{3}/{4}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  accuracy_best['train'], epoch + 1, args.epochs, i + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

        # write full summary csv
        if csv_handler['csv_file_summary_full'] is not None:
            writeCSV(
                csv_handler['csv_file_summary_full'],
                False,
                batch_time.val, # time taken
                args.session_name, # train session (e.g. all)
                epoch + 1, # epoch number current
                args.epochs, # epoch number overall
                'train', # current phase (train or val)
                (i + 1) * args.batch_size, # current processed images
                correct1_all, # correct processed images
                len(train_loader) * args.batch_size, # images overall
                (i + 1) * 100 / len(train_loader), # processed in percent
                losses.avg, # loss
                top1.avg, # accuracy in percent (top 1)
                top5.avg, # accuracy in percent (top 5)
                get_learning_rate_from_optimizer(optimizer) # learning rate
            )

    # get best accuracy (top 1)
    accuracy_best['train'] = top1.avg if top1.avg > accuracy_best['train'] else accuracy_best['train']

    # get best accuracy (top 5)
    accuracy_best['train5'] = top5.avg if top5.avg > accuracy_best['train5'] else accuracy_best['train5']

    # write summary csv
    if csv_handler['csv_file_summary'] is not None:
        writeCSV(
            csv_handler['csv_file_summary'],
            False,
            time_all, # time taken
            args.session_name, # train session
            epoch + 1, # epoch number current
            args.epochs, # epoch number overall
            'train', # current phase (train or val)
            losses.avg, # loss
            top1.avg, # accuracy in percent (top 1)
            accuracy_best['train'], # best accuracy in percent' (top 1)
            top5.avg, # accuracy in percent (top 5)
            accuracy_best['train5'], # best accuracy in percent' (top 5)
            get_learning_rate_from_optimizer(optimizer) # learning rate
        )


def validate(val_loader, model, criterion, args, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():

        end = time.time()
        correct1_all = 0
        time_all = 0
        for i, (input, target, path) in enumerate(val_loader):

            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            if not args.use_cpu:
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # compute class names
            class_names = val_loader.dataset.classes

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            correct1, correct5 = accuracy_count(output, target, topk=(1, 5))
            correct1_all += correct1

            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            time_all += batch_time.val

            if args.print_freq == 1 and args.batch_size == 1:
                pred_percent, pred_class = predicted(output, 5, class_names)
                target_class_name = class_names[int(target)]

                print('Val ({top1.avg:.3f}%/{0:.3f}%): [{1}/{2}][{3}/{4}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      '{path}\t'
                      '{correct}\t'
                      '{pred_class}\t'
                      '{pred_percent}\t'
                      '{real_class}'.format(
                          accuracy_best['val'], epoch + 1, args.epochs, i + 1, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5, path=path[0],
                          correct='True' if pred_class[0] == target_class_name else 'False',
                          pred_class=','.join([str(x) for x in pred_class]),
                          pred_percent=','.join([str(x) for x in pred_percent]),
                          real_class=target_class_name
                      )
                )

                if csv_handler['csv_file_validated'] is not None:
                    writeCSV(
                        csv_handler['csv_file_validated'],
                        False,
                        batch_time.avg,
                        path[0],
                        pred_class[0] == target_class_name,
                        target_class_name,
                        pred_class[0],
                        pred_percent[0],
                        pred_class[1],
                        pred_percent[1],
                        pred_class[2],
                        pred_percent[2],
                        pred_class[3],
                        pred_percent[3],
                        pred_class[4],
                        pred_percent[4]
                    )
            elif i % args.print_freq == 0:
                print('Val ({top1.avg:.3f}%/{0:.3f}%): [{1}/{2}][{3}/{4}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       accuracy_best['val'], epoch + 1, args.epochs, i + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))

            # write full summary csv
            if csv_handler['csv_file_summary_full'] is not None:
                writeCSV(
                    csv_handler['csv_file_summary_full'],
                    False,
                    batch_time.val,  # time taken
                    args.session_name,  # train session (e.g. all)
                    epoch + 1,  # epoch number current
                    args.epochs,  # epoch number overall
                    'val',  # current phase (train or val)
                    (i + 1) * args.batch_size, # current processed images
                    correct1_all,  # correct processed images
                    len(val_loader) * args.batch_size,  # images overall
                    (i + 1) * 100 / len(val_loader),  # processed in percent
                    losses.avg,  # loss
                    top1.avg,  # accuracy in percent (top 1)
                    top5.avg,  # accuracy in percent (top 5)
                    0 # learning rate
                )

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        # get best accuracy (top 1)
        accuracy_best['val'] = top1.avg if top1.avg > accuracy_best['val'] else accuracy_best['val']

        # get best accuracy (top 5)
        accuracy_best['val5'] = top5.avg if top5.avg > accuracy_best['val5'] else accuracy_best['val5']

        # write summary csv
        if csv_handler['csv_file_summary'] is not None:
            writeCSV(
                csv_handler['csv_file_summary'],
                False,
                time_all, # time taken
                args.session_name, # train session
                epoch + 1, # epoch number current
                args.epochs, # epoch number overall
                'val', # current phase (train or val)
                losses.avg, # loss
                top1.avg, # accuracy in percent
                accuracy_best['val'], # best accuracy in percent'
                top5.avg, # accuracy in percent
                accuracy_best['val5'], # best accuracy in percent'
                0 # learning rate
            )

    return top1.avg


def save_checkpoint(state, is_best, args):

    # build the target paths
    checkpoint_path = getFormatedPath(
        args.model_path,
        {
            'name': 'checkpoint',
            'model_name': args.arch,
            'input_size': settings['input_size'],
            'device_name': settings['device_name']
        },
        0,
        time_start,
        args.lr,
        args.momentum,
        args.batch_size,
        args.workers,
        args.weight_decay,
        args.pretrained,
        '{}_lr{}_m{}_bs{}_w{}_wd{}_{}.{}pth'
    )

    best_path = getFormatedPath(
        args.model_path,
        {
            'name': 'model_best',
            'model_name': args.arch,
            'input_size': settings['input_size'],
            'device_name': settings['device_name']
        },
        0,
        time_start,
        args.lr,
        args.momentum,
        args.batch_size,
        args.workers,
        args.weight_decay,
        args.pretrained,
        '{}_lr{}_m{}_bs{}_w{}_wd{}_{}.{}pth'
    )

    # prepare folder for saving the model
    if args.model_path is not None:
        createFolderForFile(checkpoint_path)

    torch.save(state, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, best_path)


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


def adjust_learning_rate(optimizer, epoch, args):
    """ Sets the learning rate to the initial LR by factor args.learning_rate_decrease_factor
    every args.learning_rate_decrease_after epochs"""
    lr = args.lr * (args.learning_rate_decrease_factor ** (epoch // args.learning_rate_decrease_after))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def accuracy_count(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(int(correct_k))
        return res


def predicted(output, topk, class_names):
    with torch.no_grad():
        pred_percent, pred_class = output.topk(topk, 1, True, True)

        pred_percent = pred_percent.tolist()[0]
        pred_class = pred_class.tolist()[0]

        # translate class ids into class names
        for idx, class_id in enumerate(pred_class):
            pred_class[idx] = class_names[class_id]

        return [pred_percent, pred_class]


def get_learning_rate_from_optimizer(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    main()
