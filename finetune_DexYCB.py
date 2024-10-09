import sys
import os
from typing import *
import argparse
import time
import datetime
import json
import math
from pathlib import Path

import random
import einops
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision.transforms as transforms
import timm.optim.optim_factory as optim_factory

from utils import vis_mano
import misc
from misc import NativeScalerWithGradNormCount as NativeScaler
import vit
import resnet
import lr_sched
from loss import PoseLoss

from dataset import DexYCB


def get_args_parser():
    parser = argparse.ArgumentParser('Pretraining', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter" \
                            " * "# gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch " \
                            "size under memory constraints)')

    # Dataset configure
    parser.add_argument('--background_removal', action="store_true",
                        help='Remove background to align the training and evaluation')
    parser.set_defaults(background_removal=False)

    # Model parameters
    parser.add_argument('--model', default='resnet/pose_resnet50', type=str, metavar='MODEL',
                        help='Name of model to finetune')
    parser.add_argument('--backbone_ckpt', default=None, type=str, help='Path to pre-trained " \
                            "backbone checkpoint')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # parser.add_argument('--norm_pix_loss', action='store_true',
    #                     help='Use (per-patch) normalized pixels as targets for computing loss')
    # parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gradient clipping max norm (default: None, no clipping)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--output_dir', default='./logs/debug',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./logs/debug',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) " \
                            "transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    pose_loss = PoseLoss()
    optimizer.zero_grad()

    for data_iter_step, data_item in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images = data_item['image'].to(device)
        pose_gt = data_item['pose'].to(device)
        pose_valid = data_item['valid'].to(device)[:,None]

        with torch.amp.autocast("cuda"):  # torch.cuda.amp.autocast():
            pose_pred = model(images)

        # * calculate the loss
        loss_mano = pose_loss(pose_pred, pose_gt, pose_valid) 
        loss = {'backward': loss_mano, 'loss_mano': loss_mano}

        loss_value = loss['backward'].item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        for k in loss.keys():
            loss[k] /= accum_iter
        
        # backward manually
        loss['backward'].backward()
        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        # step
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.step()

        # calulate the gradient norm for checking
        total_norm = 0.
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        for k in loss.keys():
            metric_logger.update(**{k: loss[k].item()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        reduced_values = {k: misc.all_reduce_mean(v) for k, v in loss.items()}
        if log_writer is not None and data_iter_step % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            for k, v in reduced_values.items():
                log_writer.add_scalar('train/{}'.format(k), v, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            # calculate the gradient norm
            log_writer.add_scalar('grad_norm', total_norm, epoch_1000x)
        if log_writer is not None and (data_iter_step // accum_iter) % 30 == 0:
            ''' Visulizaing the reconstructing result
            '''
            mano_vis = vis_mano(pose_gt.detach().cpu(),
                                einops.rearrange(pose_pred.detach().cpu(), 'b j d -> b (j d)'),
                                data_item['shape'].detach().cpu(),
                                data_item['shape'].detach().cpu(), 'right')
            log_writer.add_image("mano_mesh",
                                 einops.rearrange(torch.from_numpy(mano_vis), "h w c -> c h w"),
                                 epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    # writing trainig config to file
    with open(os.path.join(args.output_dir, "config.txt"), mode="w", encoding="utf-8") as f:
        f.write("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # seed fixed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    dataset_train = DexYCB("s0", "train")
    print(f"total length = {len(dataset_train)}")

    if True:  # ? args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sample_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    data_loader_train = DataLoader(  # torch.utils.data.DataLoader
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # * Define the model
    # define the model
    model_str = args.model
    model_class, model_arch = model_str.split('/')
    if model_class == 'vit':
        model = vit.__dict__[model_arch](norm_pix_loss=args.norm_pix_loss)
    elif model_class == 'resnet':
        model = resnet.__dict__[model_arch](predict_mano=True,
                                            backbone_ckpt=args.backbone_ckpt)
    else:
        assert False, "model not supported: %s" % model_str

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed: 
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # configure ddp training
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        # log epoch status
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
