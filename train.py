import random
import os
import shutil
import json
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from toolbox.datasets import get_dataset
from toolbox.log import get_logger
from toolbox.models import get_model
from toolbox.loss import get_loss
from toolbox.metrics import averageMeter, runningScore
from toolbox.utils import tensor_classes_to_RGBs, color_map


def run(cfg, logger, writer):
    logger.info(f'Conf | use dataset {cfg["dataset"]}')
    logger.info(f'Conf | use augmentation {cfg["augmentation"]}')

    # 输入尺寸
    cfg['image_size'] = (cfg['image_h'], cfg['image_w'])
    logger.info(f'Conf | use image size {cfg["image_size"]}')

    # 获取训练集和验证集
    trainset, valset, *testset = get_dataset(cfg)

    logger.info(f'Conf | use batch_size {cfg["batch_size"]}')
    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    # 模型
    logger.info(f'Conf | use model {cfg["model_name"]}')
    model = get_model(cfg)

    # 是否多gpu训练
    gpu_ids = [int(i) for i in list(cfg['gpu_ids'])]
    logger.info(f'Conf | use GPU {gpu_ids}')
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(cfg["device"])

    # 优化器 & 学习率衰减 可根据情况修改
    logger.info(
        f'Conf | use optimizer SGD, lr={cfg["lr"]}, momentum={cfg["momentum"]}, weight_decay={cfg["weight_decay"]}')
    logger.info(f'Conf | use poly learning rate policy power={cfg["power"]}.')
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epoch']) ** cfg['power'])

    # 损失函数 & 类别权重平衡 & 训练时包含unlabel
    logger.info(f'Conf | use loss function {cfg["loss"]}')
    criterion = get_loss(cfg, weight=trainset.class_weight).to(cfg['device'])

    # 指标 包含unlabel
    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()
    running_metrics_val = runningScore(cfg['n_classes'])

    iter = 0
    best_val_miou_meter = -100  # 保存验证集miou最好模型

    logger.info(f'Conf | use epoch {cfg["epoch"]}')

    # 每个epoch迭代循环
    for ep in range(cfg['epoch']):

        # training
        model.train()
        scheduler.step(ep)

        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            iter += 1
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            image = sample['image'].to(cfg['device'])
            label = sample['label'].to(cfg['device'])

            predict = model(image)
            loss = criterion(predict, label)
            ####################################################

            train_loss_meter.update(loss.item())

            loss.backward()
            optimizer.step()

            # print every 50 iters
            if (i + 1) % 50 == 0:
                logger.info(f'Iter | [{ep + 1:3d}/{cfg["epoch"]}] [{i + 1:4d}/'
                            f'{len(train_loader)}] [{iter:10d}] train loss is {train_loss_meter.avg: .8f}')
                cmap = trainset.cmap if hasattr(trainset, 'cmap') else None
                grid_image = make_grid(image[:3].clone().cpu(), 3, normalize=True)
                writer.add_image('image', grid_image, iter)
                grid_image = make_grid(tensor_classes_to_RGBs(torch.max(predict[:3], 1)[1], cfg['n_classes'], cmap), 3,
                                       normalize=False,
                                       range=(0, 255))
                writer.add_image('Predicted label', grid_image, iter)
                grid_image = make_grid(tensor_classes_to_RGBs(label[:3], cfg['n_classes'], cmap), 3, normalize=False,
                                       range=(0, 255))
                writer.add_image('Groundtruth label', grid_image, iter)
                writer.add_scalar('train CrossEntropyLoss', loss.data, global_step=iter)
                writer.add_scalar('Learning rate', scheduler.get_lr()[0], global_step=iter)

        # val
        with torch.no_grad():
            model.eval()
            val_loss_meter.reset()
            running_metrics_val.reset()

            for i, sample in enumerate(val_loader):
                ################### val edit #######################
                image = sample['image'].to(cfg['device'])
                label = sample['label'].to(cfg['device'])

                predict = model(image)
                loss = criterion(predict, label)
                ####################################################

                val_loss_meter.update(loss.item())

                predict = predict.max(1)[1].cpu().numpy()  # [batch_size, h, w]
                label = label.cpu().numpy()  # [batch_size, 1, h, w] -> [batch_size, h, w]
                running_metrics_val.update(label, predict)

            writer.add_scalars('loss / epoch', {
                'train_loss': train_loss_meter.avg,
                'val_loss': val_loss_meter.avg
            }, ep + 1)

            logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] loss:train/val='
                        f'{train_loss_meter.avg:.8f}/{val_loss_meter.avg:.8f}')

            score, class_iou = running_metrics_val.get_scores()

            for key, value in score.items():
                logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] {key}{value}')

            writer.add_scalars('metrics', score, ep + 1)

            # 如果结果最好 保存模型
            if score["mIou: "] > best_val_miou_meter:
                best_val_miou_meter = score["mIou: "]
                save_state_dict = model.module.state_dict() if len(gpu_ids) > 1 else model.state_dict()
                torch.save(save_state_dict, os.path.join(cfg['logdir'], 'best_val_miou.pth'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/ade20k_drn_c_26.json",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    # 训练的各种记录的保存目录
    logdir = f'run/{cfg["dataset"]}/{time.strftime("%Y-%m-%d-%H-%M")}-{random.randint(1000, 10000)}'  # the same time as log + random id
    os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    writer = SummaryWriter(log_dir=logdir)

    logger.info(f'Conf | use logdir {logdir}')

    cfg['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg['logdir'] = logdir

    run(cfg, logger, writer)

    writer.close()
