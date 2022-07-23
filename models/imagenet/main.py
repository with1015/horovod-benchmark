import argparse
import os
import random
import shutil
import time
import warnings
import datetime

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

import horovod.torch as hvd
import grad_hook

hvd.init()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

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
parser.add_argument('--num-minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--no-validate', dest='no_validate', action='store_true',
                    help="No validation")
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
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
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--checkpoint-path', type=str,
                    default=None,
                    help='checkpoint path')
parser.add_argument('--lr-scaling', action='store_true',
                    help="LR linear scaling rule")
parser.add_argument('--profile-dir', type=str, default=None,
                    help='Profile json file. '
                         'If the file path is set up, profiler runs.')
parser.add_argument('--perf-breakdown', action='store_true',
                    help="Flags for performance breakdown")
parser.add_argument('--check-sparsity', action='store_true',
                    help="Check sparse rate")

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.perf_breakdown:
        print('[INFO] CUDA LAUNCH BLOCKING - synchronous CPU-GPU runtime for breakdown')
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.num_minibatches is not None:
        torch.cuda.empty_cache()

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

    ngpus_per_node = torch.cuda.device_count()
    args.global_batch_size = args.batch_size * hvd.size()
    main_worker(args)


def main_worker(args):
    global best_acc1
    args.gpu = hvd.local_rank()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)

    print('Global Batch Size:', args.global_batch_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('inception'):
            model = models.inception_v3(transform_input=True, aux_logits=False)
        else:
            model = models.__dict__[args.arch]()


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters())

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # Reference: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
    if (args.arch == 'resnet50' and args.global_batch_size > 256 and
        not args.lr_scaling and not args.num_minibatches):
        args.lr_scaling = True
        print('LR linear scaling rule is applied '
              'when training ResNet-50 with the global batch size > 256')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = False

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.arch.startswith('inception'):
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=hvd.size(),
                                                                    rank=hvd.rank())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.num_minibatches is not None:
        print('Number of mini-batches:', args.num_minibatches)
        args.start_epoch = 0
        args.epochs = 1
        print('Start epoch, epochs:', args.start_epoch, args.epochs)

    total_train_time = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        epoch_start_time = time.time()
        if args.profile_dir:
            print('[INFO] Run with pytorch profiler. It may degrade the performance.')
            profile_train(train_loader, model, criterion, optimizer, epoch, args)
        else:
            train(train_loader, model, criterion, optimizer, epoch, args)
        epoch_time = int(time.time() - epoch_start_time)
        print('Epoch time:', epoch_time)
        total_train_time += epoch_time

        is_best = False
        if not args.no_validate:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

        if args.checkpoint_path and args.epochs > 1:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

                checkpoint_dir = os.path.dirname(args.checkpoint_path)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, args.checkpoint_path)
    print('Total epoch time (sec):', total_train_time)
    print('Total epoch time:', datetime.timedelta(seconds=total_train_time))


def profile_train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    #    profile_memory=True
    wait, warmup, active, repeat = 10, 1, 10, 1
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_dir),
        record_shapes=True,
        with_stack=True,
        ) as prof:
        for i, (images, target) in enumerate(train_loader):
            if i >= (wait + warmup + active) * repeat:
                print(f'[INFO] step: {i} - profiler finishes.')
                break
            if args.num_minibatches is not None and i > args.num_minibatches:
                break

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # compute gradient and do SGD step
            if not args.no_optimizer:
                optimizer.zero_grad()
            loss.backward()
            if not args.no_optimizer:
                optimizer.step()

            prof.step()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    forward_time = AverageMeter('Forward', ':6.3f')
    gradient_time = AverageMeter('Gradient', ':6.3f')
    update_time = AverageMeter('Update', ':6.3f')
    backward_time = AverageMeter('Backward', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    if args.num_minibatches is not None:
        num_batches = args.num_minibatches
    else:
        num_batches = len(train_loader)

    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time,
         forward_time, backward_time, gradient_time, update_time,
         losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    if args.num_minibatches is not None:
        print('Warm-up training start!')
        for i in range(10):
            if args.arch.startswith('inception'):
                input_shape = [args.batch_size, 3, 299, 299]
            else:
                input_shape = [args.batch_size, 3, 224, 224]
            dummy_input = torch.randn(*input_shape).to(args.gpu, non_blocking=True)

            model(dummy_input)

    torch.cuda.synchronize()
    print('Warm-up training end!')
    is_outlier = False
    end = time.time()

    sparse_rate = []
    if args.check_sparsity:
        b_hook = grad_hook.hook_to_model(model, backward=True)

    for i, (images, target) in enumerate(train_loader):
        if args.num_minibatches is not None and i > args.num_minibatches:
            break

        # measure data loading time
        if i >= 10:
            loading_time = time.time() - end
            if args.workers > 1 and loading_time >= 1 and args.num_minibatches is not None:
                print(f'[WARNING] loading outlier with loading workers = {args.workers}: {loading_time:6.5f}')
                is_outlier = True
            if not is_outlier:
                data_time.update(loading_time)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if i >= 10 and not is_outlier:
            start = time.time()
        output = model(images)
        loss = criterion(output, target)
        if i >= 10 and not is_outlier:
            forward_time.update(time.time() - start)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        if i >= 10 and not is_outlier:
            start = time.time()
            bwd_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        if i >= 10 and not is_outlier:
            gradient_time.update(time.time() - start)
            start = time.time()
        optimizer.step()


        if i >= 10 and args.check_sparsity:
            for hook in b_hook:
                if len(hook.inputs) > 0:
                    for idx in range(len(hook.inputs)):
                        if hook.inputs[idx] == None:
                            continue
                        cnt = hook.inputs[idx].numel() - hook.inputs[idx].nonzero().size(0)
                        rate = float(cnt) / float(hook.inputs[idx].numel())
                        sparse_rate.append(rate)


        if i >= 10 and not is_outlier:
            update_time.update(time.time() - start)
            backward_time.update(time.time() - bwd_start)

        # measure elapsed time
        if i >= 10 and not is_outlier:
            batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0:
            progress.display(i)
            if i >= 10 and args.check_sparsity:
                if len(sparse_rate) == 0:
                    continue
                print("[DEBUG] average sparse rate:", sum(sparse_rate) / len(sparse_rate))
                sparse_rate = []



def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        checkpoint_dir = os.path.dirname(filename)
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    if 'resnet' in args.arch and args.lr_scaling and args.global_batch_size > 256:
        warmup_epochs = 5
        print('Apply lr linear scaling rule')
        # Graudal warmup
        if epoch < warmup_epochs:
            lr = args.lr * ((epoch + 1) / warmup_epochs)
        # Linear Scaling Rule
        else:
            linear_scale_factor = args.global_batch_size / 256
            lr = args.lr * linear_scale_factor * (0.1 ** (epoch // 30))
    else:
        lr = args.lr * (0.1 ** (epoch // 30))

    print('Learning rate {} at epoch {}'.format(lr, epoch))
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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
