import os
import shutil
import json
import time

from apex import amp
import apex

import numpy as np
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader

from toolbox import MscCrossEntropyLoss
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt

torch.manual_seed(123)
cudnn.benchmark = True


def run(args):

    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}'
    args.logdir = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    if args.local_rank == 0:
        logger.info(f'Conf | use logdir {logdir}')

    model = get_model(cfg)
    trainset, *testset = get_dataset(cfg)
    device = torch.device('cuda')

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.local_rank == 0:
            print(f"WORLD_SIZE is {os.environ['WORLD_SIZE']}")

    train_sampler = None
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()

        model = apex.parallel.convert_syncbn_model(model)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    model.to(device)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=(train_sampler is None),
                              num_workers=cfg['num_workers'], pin_memory=True, sampler=train_sampler)

    params_list = model.parameters()
    optimizer = torch.optim.Adam(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    if args.distributed:
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

    # class weight 计算
    if hasattr(trainset, 'class_weight'):
        print('using classweight in dataset')
        class_weight = trainset.class_weight
    else:
        classweight = ClassWeight(cfg['class_weight'])
        class_weight = classweight.get_weight(train_loader, cfg['n_classes'])
    class_weight = torch.from_numpy(class_weight).float().to(device)
    class_weight[cfg['id_unlabel']] = 0

    # 损失函数 & 类别权重平衡 & 训练时ignore unlabel
    criterion = MscCrossEntropyLoss(weight=class_weight).to(device)

    # 指标 包含unlabel
    train_loss_meter = averageMeter()

    # 每个epoch迭代循环
    for ep in range(cfg['epochs']):
        if args.distributed:
            train_sampler.set_epoch(ep)

        # training
        model.train()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            predict = model(image)
            loss = criterion(predict, label)
            ####################################################

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            if args.distributed:
                reduced_loss = loss.clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss /= args.world_size
            else:
                reduced_loss = loss
            train_loss_meter.update(reduced_loss.item())

        scheduler.step(ep)

        if args.local_rank == 0:
            logger.info(f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] train loss={train_loss_meter.avg:.5f}')
            save_ckpt(logdir, model)

    save_ckpt(logdir, model)


if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cityscape_drn_c_26.json",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--opt_level",
        type=str,
        default='O1',
    )

    args = parser.parse_args()

    run(args)


