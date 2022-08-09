import argparse
import datetime
from copy import deepcopy

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
import horovod.torch as hvd

hvd.init()

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
parser.add_argument('--data-config', type=str, default='data/coco2017.data', help='*.data path')
parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
#parser.add_argument('--img-size', nargs='+', type=int, default=[320], help='[min_train, max-train, test]')
parser.add_argument('--img-size', nargs='+', type=int, default=[320, 320, 320], help='[min_train, max-train, test]')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--notest', action='store_true', help='only test final epoch')
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='initial weights path')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
parser.add_argument('--freeze-layers', action='store_true', help='Freeze non-output layers')
# [GC] Add args
parser.add_argument('data', metavar='DIR', help='path to dataset directoty')
parser.add_argument('--num-minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--profile-dir', type=str, default=None,
                    help='Profile dir path'
                    'If the file path is set up, profiler runs.')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume-checkpoint-file', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none) for resume')
parser.add_argument('--checkpoint-dir', type=str, default=None,
                    help='Checkpoint dir path to save')
parser.add_argument('--debug', action='store_true', help="Print debug message")
parser.add_argument('--is-plot', action='store_true', help="Plot results or images")
parser.add_argument('--accumulate-steps', default=1, type=int, help='Gradient accumulation steps')
parser.add_argument('--save-json', action='store_true', help="Save json file for test")
parser.add_argument('--loss-verbose', action='store_true',
                    help="Verbose mean loss - GIoU, objectness, cls, total")
parser.add_argument('--perf-breakdown', action='store_true',
                    help="Flags for performance breakdown")
# [GC] Distributed training
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to run single-GPU training.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# Experimental args
parser.add_argument('--no-optimizer', action='store_true',
                    help="Run without optimizer to measure computation only")
parser.add_argument('--fuse-conv-and-bn', action='store_true',
                    help="Fuse conv layer and batch norm layer in model")


def main():
    args = parser.parse_args()
    #print(f"Command line arguments: {args}")


    if args.perf_breakdown:
        print('[INFO] CUDA LAUNCH BLOCKING - synchronous CPU-GPU runtime for breakdown')
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


    if args.num_minibatches is not None:
        torch.cuda.empty_cache()

    if args.gpu is not None:
        print('[INFO] You have chosen a specific GPU. This will completely '
              'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    args.global_batch_size = args.batch_size * hvd.size()
    main_worker(args)


def main_worker(args):
    args.gpu = hvd.local_rank()
    device = args.gpu
    print("Use GPU: {} for training".format(device))
    torch.cuda.set_device(device)

    print("Global Batch Size:",args.global_batch_size)
    # Hyperparameters
    args.hyp = {'giou': 3.54,  # giou loss gain
           'cls': 37.4,  # cls loss gain
           'cls_pw': 1.0,  # cls BCELoss positive_weight
           'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
           'obj_pw': 1.0,  # obj BCELoss positive_weight
           'iou_t': 0.20,  # iou training threshold
           'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': 0.0005,  # final learning rate (with cos scheduler)
           'momentum': 0.937,  # SGD momentum
           'weight_decay': 0.0005,  # optimizer weight decay
           'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 1.98 * 0,  # image rotation (+/- deg)
           'translate': 0.05 * 0,  # image translation (+/- fraction)
           'scale': 0.05 * 0,  # image scale (+/- gain)
           'shear': 0.641 * 0}  # image shear (+/- deg)

    # Print focal loss if gamma > 0
    if args.hyp['fl_gamma']:
        print('Using FocalLoss(gamma=%g)' % args.hyp['fl_gamma'])

    #check_git_status()
    #args.cfg = check_file(args.cfg)  # check file
    #args.data_config = check_file(args.data_config)  # check file

    args.img_size.extend([args.img_size[-1]] * (3 - len(args.img_size)))  # extend to 3 sizes (min, max, test)

    cfg = args.cfg
    data_config = args.data_config
    epochs = args.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = args.batch_size
    #accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    #print(f'[DEBUG] initial accumulate: {accumulate}')
    imgsz_min, imgsz_max, imgsz_test = args.img_size  # img sizes (min, max, test)

    # Image Sizes
    args.grid_size = 32  # (pixels) grid size
    assert math.fmod(imgsz_min, args.grid_size) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, args.grid_size)
    #args.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    args.multi_scale = True
    print('[INFO] multi-scale training has been done in original YOLO paper')
    if args.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        args.grid_min, args.grid_max = imgsz_min // args.grid_size, imgsz_max // args.grid_size
        imgsz_min, imgsz_max = int(args.grid_min * args.grid_size), int(args.grid_max * args.grid_size)
    img_size = imgsz_max  # initialize with max size

    # Configure run
    init_seeds()
    data_dict = parse_data_cfg(data_config)
    train_path = os.path.join(args.data, data_dict['train'])
    test_path = os.path.join(args.data, data_dict['valid'])
    print(f'[INFO] train_path: {train_path}')
    print(f'[INFO] test_path: {test_path}')
    nc = 1 if args.single_cls else int(data_dict['classes'])  # number of classes
    args.hyp['cls'] *= nc / 80  # update coco-tuned args.hyp['cls'] to current dataset

    # Initialize model
    if args.fuse_conv_and_bn:
        print('[INFO] Fuse model!')
        model = Darknet(cfg)
    else:
        model = Darknet(cfg).to(device)
    model = model.cuda()
    model_for_ema = deepcopy(model)

    
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if args.adam:
        optimizer = torch.optim.Adam(pg0, lr=args.hyp['lr0'])

    else:
        optimizer = torch.optim.SGD(pg0, lr=args.hyp['lr0'], momentum=args.hyp['momentum'], nesterov = True)
    optimizer.add_param_group({'params' : pg1, 'weight_decay' : args.hyp['weight_decay']})
    optimizer.add_param_group({'params' : pg2})
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    optimizer = hvd.DistributedOptimizer(optimizer,
                                        named_parameters=model.named_parameters())

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    start_epoch = 0
    best_fitness = 0.0
    attempt_download(args.weights)
    if args.resume_checkpoint_file:
        load_checkpoint(model, optimizer, epochs, start_epoch, device, best_fitness, args)

    if args.weights:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, args.weights)

    # Fusion must be applied on CPU and after loading pre-trained weights
    if args.fuse_conv_and_bn:
        print('[EXPERIMENT] Fuse model!')
        model.fuse()
        model = model.to(device)

    if args.freeze_layers:
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if isinstance(module, YOLOLayer)]
        freeze_layer_indices = [x for x in range(len(model.module_list)) if
                                (x not in output_layer_indices) and
                                (x - 1 not in output_layer_indices)]
        for idx in freeze_layer_indices:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    args.lr_schedule_fn = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=args.lr_schedule_fn)
    scheduler.last_epoch = start_epoch - 1  # see link below
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822

    # Initialize distributed training

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=args.hyp,  # augmentation hyperparameters
                                  rect=args.rect,  # rectangular training
                                  cache_images=args.cache_images,
                                  single_cls=args.single_cls)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=hvd.size(),
                                                                    rank=hvd.rank())

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = 4 #min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of data loading workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=(train_sampler is None and not args.rect),  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn,
                                             sampler=train_sampler)

    # Testloader
    test_batch_size = 4 if args.batch_size > 4 else 1
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                 hyp=args.hyp,
                                                                 rect=True,
                                                                 cache_images=args.cache_images,
                                                                 single_cls=args.single_cls),
                                             batch_size=test_batch_size,
                                             num_workers=nw,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = args.hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights

    # Model EMA
    ema = torch_utils.ModelEMA(model_for_ema)

    # Start training
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)

    if args.num_minibatches is not None:
        print('[INFO] Number of mini-batches:', args.num_minibatches)
        print('[INFO] If args.num_minibatches is set, args.epochs is set to 1')
        epochs = 1

    total_train_time = 0
    for epoch in range(start_epoch, epochs):

        epoch_start_time = time.time()
        if args.profile_dir:
            print('[INFO] Run with pytorch profiler. It may degrade the performance.')
            profile_train(dataloader, dataset, model, ema, optimizer, device, epoch, args)
            break
        elif args.no_optimizer:
            print('[EXPERIMENT] Run without optimizer to measure computation only')
            run_without_optimizer(dataloader, dataset, model, ema, optimizer, device, epoch, args)
            break
        else:
            train(dataloader, dataset, model, ema, optimizer, device, epoch, args)
        epoch_time = int(time.time() - epoch_start_time)
        print('Epoch time:', epoch_time)
        total_train_time += epoch_time

        # Update scheduler
        scheduler.step()

        # Process epoch results
        ema.update_attr(model)
        is_final_epoch = (epoch + 1 == epochs)

        # Calculate mAP
        if not args.notest:
            is_coco = any([x in data_config for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            num_batches = len(dataloader)
            ni =  num_batches * (epoch + 1)
            n_burn = max(3 * num_batches, 500)
            is_multi_label = (ni > n_burn)
            results, maps = test.test(cfg,
                                    data_config,
                                    batch_size=test_batch_size,
                                    imgsz=imgsz_test,
                                    model=ema.ema,
                                    save_json=(is_final_epoch and is_coco and args.save_json),
                                    single_cls=args.single_cls,
                                    dataloader=testloader,
                                    multi_label=is_multi_label)

        # Update best mAP
        # fitness_i = weighted combination of [P, R, mAP, F1]
        fi = fitness(np.array(results).reshape(1, -1))
        if fi > best_fitness:
            best_fitness = fi

        # Save ckpt
        is_save = (not args.nosave and args.checkpoint_dir)
        if is_save:
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            filename = os.path.join(args.checkpoint_dir, 'last_ckpt.pt')
            is_best = ((best_fitness == fi) and not is_final_epoch)
            state = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                'optimizer': None if is_final_epoch else optimizer.state_dict()
            }
            save_checkpoint(state, filename, is_best)

    # end training
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    print('Total epoch time (sec):', total_train_time)
    print('Total epoch time:', datetime.timedelta(seconds=total_train_time))
    #dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()


def profile_train(dataloader, dataset, model, ema, optimizer, device, epoch, args):
    if args.num_minibatches is not None:
        num_batches = args.num_minibatches
    else:
        num_batches = len(dataloader)
    n_burn = max(3 * num_batches, 500)  # burn-in iterations, max(3 epochs, 500 iterations)

    # Update image weights (optional)
    if dataset.image_weights:
        nc = model.nc
        maps = np.zeros(nc)  # mAP per class
        w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
        image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
        dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

    # switch to train mode
    model.train()
    wait, warmup, active, repeat = 10, 1, 1, 1
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_dir),
        record_shapes=True,
        with_stack=True,
        ) as prof:
        for i, (imgs, targets, paths, _) in enumerate(dataloader):
            if i >= (wait + warmup + active) * repeat:
                print(f'[INFO] step: {i} - profiler finishes.')
                break
            if args.num_minibatches is not None and i > args.num_minibatches:
                break
            ni = i + num_batches * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Burn-in
            if ni <= n_burn:
                #print(f'[DEBUG] condition ni <= n_burn {ni} <= {n_burn}')
                xi = [0, n_burn]  # x interp
                model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                #args.accumulate_steps = max(1, np.interp(ni, xi, [1, 64 / args.global_batch_size]).round())
                #accumulate = max(1, np.interp(ni, xi, [1, 64 / args.batch_size]).round())
                #print(f'[DEBUG] accumulate: {accumulate}')
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * args.lr_schedule_fn(epoch)])
                    x['weight_decay'] = np.interp(ni, xi, [0.0, args.hyp['weight_decay'] if j == 1 else 0.0])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, args.hyp['momentum']])

            # Multi-Scale
            if args.multi_scale:
                if ni / args.accumulate_steps % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                    img_size = random.randrange(args.grid_min, args.grid_max + 1) * args.grid_size
                scale_factor = img_size / max(imgs.shape[2:])  # scale factor
                if scale_factor != 1:
                    # new shape (stretched to 32-multiple)
                    new_shape = [math.ceil(x * scale_factor / args.grid_size) * args.grid_size for x in imgs.shape[2:]]
                    imgs = F.interpolate(imgs, size=new_shape, mode='bilinear', align_corners=False)

            # Forward
            if not args.no_optimizer:
                optimizer.zero_grad()
            pred = model(imgs)
            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return

            # Backward
            #loss *= batch_size / 64  # scale loss
            loss *= args.global_batch_size / 64  # scale loss
            #print(f'[DEBUG] loss: {loss}')
            loss.backward()

            # Optimize
            if not args.no_optimizer:
                optimizer.step()
            ema.update(model)

            prof.step()


def train(dataloader, dataset, model, ema, optimizer, device, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.5f')
    prep_time = AverageMeter('Prep', ':6.5f')
    forward_time = AverageMeter('Forward', ':6.5f')
    gradient_time = AverageMeter('Gradient', ':6.5f')
    update_time = AverageMeter('Update', ':6.5f')
    backward_time = AverageMeter('Backward', ':6.5f')
    ema_time = AverageMeter('EMA', ':6.5f')
    losses = AverageMeter('Loss', ':.4e')

    if args.num_minibatches is not None:
        num_batches = args.num_minibatches
    else:
        num_batches = len(dataloader)
    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, prep_time,
         forward_time, backward_time, gradient_time, update_time, ema_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    n_burn = max(3 * num_batches, 500)  # burn-in iterations, max(3 epochs, 500 iterations)

    # Update image weights (optional)
    if dataset.image_weights:
        nc = model.nc
        maps = np.zeros(nc)  # mAP per class
        w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
        image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
        dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

    # switch to train mode
    model.train()

    if args.loss_verbose:
        mloss = torch.zeros(4).to(device)  # mean losses
    end = time.time()
    IS_BREAKDOWN = True #False
    print(len(dataloader))
    for i, (imgs, targets, paths, _) in enumerate(dataloader):
        if args.num_minibatches is not None and i > args.num_minibatches:
            break
        ni = i + num_batches * epoch  # number integrated batches (since train start)

        # measure data loading time
        #if i > 10:
        #    data_time.update(time.time() - end)

        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        #print(f'[DEBUG] imgs shape: {imgs.shape}')

        if i >= 10 and IS_BREAKDOWN:
            data_time.update(time.time() - end)

        if i >= 10 and IS_BREAKDOWN:
            start = time.time()

        # Burn-in
        if ni <= n_burn:
            #print(f'[DEBUG] condition ni <= n_burn {ni} <= {n_burn}')
            xi = [0, n_burn]  # x interp
            model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            args.accumulate_steps = max(1, np.interp(ni, xi, [1, 64 / args.global_batch_size]).round())
            #accumulate = max(1, np.interp(ni, xi, [1, 64 / args.batch_size]).round())
            #print(f'[DEBUG] accumulate: {accumulate}')
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * args.lr_schedule_fn(epoch)])
                x['weight_decay'] = np.interp(ni, xi, [0.0, args.hyp['weight_decay'] if j == 1 else 0.0])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [0.9, args.hyp['momentum']])

        # Multi-Scale
        if args.multi_scale:
            if ni / args.accumulate_steps % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                img_size = random.randrange(args.grid_min, args.grid_max + 1) * args.grid_size
            scale_factor = img_size / max(imgs.shape[2:])  # scale factor
            if scale_factor != 1:
                # new shape (stretched to 32-multiple)
                new_shape = [math.ceil(x * scale_factor / args.grid_size) * args.grid_size for x in imgs.shape[2:]]
                imgs = F.interpolate(imgs, size=new_shape, mode='bilinear', align_corners=False)

        if i >= 10 and IS_BREAKDOWN:
            prep_time.update(time.time() - start)

        # Forward
        if i >= 10 and IS_BREAKDOWN:
            start = time.time()
        optimizer.zero_grad()
        pred = model(imgs)

        # Compute loss
        loss, loss_items = compute_loss(pred, targets, model)
        """
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss_items)
            return
        """

        if i >= 10 and IS_BREAKDOWN:
            forward_time.update(time.time() - start)

        # Backward
        if i >= 10 and IS_BREAKDOWN:
            start = time.time()
            bwd_start = time.time()
        #loss *= batch_size / 64  # scale loss
        loss *= args.global_batch_size / 64  # scale loss
        #print(f'[DEBUG] loss: {loss}')
        loss.backward()
        if i >= 10 and IS_BREAKDOWN:
            gradient_time.update(time.time() - start)
            start = time.time()

        losses.update(loss.item())

        # Optimize
        optimizer.step()
        if i >= 10 and IS_BREAKDOWN:
            update_time.update(time.time() - start)
            backward_time.update(time.time() - bwd_start)
        if i >= 10 and IS_BREAKDOWN:
            start = time.time()
        ema.update(model)
        if i >= 10 and IS_BREAKDOWN:
            ema_time.update(time.time() - start)

        # measure elapsed time
        if i >= 10:
            batch_time.update(time.time() - end)
        end = time.time()

        if i >= 0 and i % args.print_freq == 0:
            progress.display(i)

            if args.loss_verbose:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                print('[Mean loss - GIoU, obj, cls, total] ', *mloss)

        """
        # Print
        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
        pbar.set_description(s)
        """

"""
Method for only experiment
"""
def run_without_optimizer(dataloader, dataset, model, ema, optimizer, device, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.5f')
    prep_time = AverageMeter('Prep', ':6.5f')
    forward_time = AverageMeter('Forward', ':6.5f')
    gradient_time = AverageMeter('Gradient', ':6.5f')
    update_time = AverageMeter('Update', ':6.5f')
    backward_time = AverageMeter('Backward', ':6.5f')
    ema_time = AverageMeter('EMA', ':6.5f')
    losses = AverageMeter('Loss', ':.4e')

    if args.num_minibatches is not None:
        num_batches = args.num_minibatches
    else:
        num_batches = len(dataloader)
    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, prep_time,
         forward_time, backward_time, gradient_time, update_time, ema_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    n_burn = max(3 * num_batches, 500)  # burn-in iterations, max(3 epochs, 500 iterations)

    # Update image weights (optional)
    if dataset.image_weights:
        nc = model.nc
        maps = np.zeros(nc)  # mAP per class
        w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
        image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
        dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

    # switch to train mode
    model.train()

    if args.loss_verbose:
        mloss = torch.zeros(4).to(device)  # mean losses
    end = time.time()
    IS_BREAKDOWN = True #False
    for i, (imgs, targets, paths, _) in enumerate(dataloader):
        if args.num_minibatches is not None and i > args.num_minibatches:
            break
        ni = i + num_batches * epoch  # number integrated batches (since train start)

        # measure data loading time
        #if i > 10:
        #    data_time.update(time.time() - end)

        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        #print(f'[DEBUG] imgs shape: {imgs.shape}')

        if i >= 10 and IS_BREAKDOWN:
            data_time.update(time.time() - end)

        if i >= 10 and IS_BREAKDOWN:
            start = time.time()

        # Burn-in
        if ni <= n_burn:
            #print(f'[DEBUG] condition ni <= n_burn {ni} <= {n_burn}')
            xi = [0, n_burn]  # x interp
            model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            args.accumulate_steps = max(1, np.interp(ni, xi, [1, 64 / args.global_batch_size]).round())
            #accumulate = max(1, np.interp(ni, xi, [1, 64 / args.batch_size]).round())
            #print(f'[DEBUG] accumulate: {accumulate}')
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * args.lr_schedule_fn(epoch)])
                x['weight_decay'] = np.interp(ni, xi, [0.0, args.hyp['weight_decay'] if j == 1 else 0.0])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [0.9, args.hyp['momentum']])

        # Multi-Scale
        if args.multi_scale:
            if ni / args.accumulate_steps % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                img_size = random.randrange(args.grid_min, args.grid_max + 1) * args.grid_size
            scale_factor = img_size / max(imgs.shape[2:])  # scale factor
            if scale_factor != 1:
                # new shape (stretched to 32-multiple)
                new_shape = [math.ceil(x * scale_factor / args.grid_size) * args.grid_size for x in imgs.shape[2:]]
                imgs = F.interpolate(imgs, size=new_shape, mode='bilinear', align_corners=False)

        if i >= 10 and IS_BREAKDOWN:
            prep_time.update(time.time() - start)

        # Forward
        if i >= 10 and IS_BREAKDOWN:
            start = time.time()
        pred = model(imgs)

        # Compute loss
        loss, loss_items = compute_loss(pred, targets, model)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss_items)
            return

        if i >= 10 and IS_BREAKDOWN:
            forward_time.update(time.time() - start)

        # Backward
        if i >= 10 and IS_BREAKDOWN:
            start = time.time()
            bwd_start = time.time()
        #loss *= batch_size / 64  # scale loss
        loss *= args.global_batch_size / 64  # scale loss
        #print(f'[DEBUG] loss: {loss}')
        loss.backward()
        if i >= 10 and IS_BREAKDOWN:
            gradient_time.update(time.time() - start)
            start = time.time()

        losses.update(loss.item())

        if i >= 10 and IS_BREAKDOWN:
            update_time.update(time.time() - start)
            backward_time.update(time.time() - bwd_start)
        if i >= 10 and IS_BREAKDOWN:
            start = time.time()
        ema.update(model)
        if i >= 10 and IS_BREAKDOWN:
            ema_time.update(time.time() - start)

        # measure elapsed time
        if i >= 10:
            batch_time.update(time.time() - end)
        end = time.time()

        if i >= 0 and i % args.print_freq == 0:
            progress.display(i)

            if args.loss_verbose:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                print('[Mean loss - GIoU, obj, cls, total] ', *mloss)


def load_checkpoint(model, optimizer, epochs, start_epoch, device, best_fitness, args):
    if os.path.isfile(args.resume_checkpoint_file):
        print("=> loading checkpoint '{}'".format(args.resume_checkpoint_file))
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        ckpt = torch.load(args.resume_checkpoint_file, map_location=device)

        # load model
        try:
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (args.weights, args.cfg, args.weights)
            raise KeyError(s) from e

        # load optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (args.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


def save_checkpoint(state, filename, is_best):
    # Save the last and best checkpoint file
    print(f'=> saving the last checkpoint: {filename}')
    torch.save(state, filename)
    if is_best:
        print(f'=> saving the best checkpoint: {filename}')
        checkpoint_dir = os.path.dirname(filename)
        best_filename = os.path.join(checkpoint_dir, 'best_ckpt.pt')
        torch.save(state, best_filename)


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


if __name__ == "__main__":
    main()
