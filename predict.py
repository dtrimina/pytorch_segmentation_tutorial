import os
import json
import time
from pprint import pprint
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as t

from toolbox.datasets import get_dataset
from toolbox.models import get_model
from toolbox.metrics import averageMeter, runningScore
from toolbox.utils import class_to_RGB


def predict(dataset, runid, use_pth='best_val_loss.pth', target_size=None, save_predict=False):
    assert use_pth in ['best_val_loss.pth']

    logdir = f'run/{dataset}/{runid}'
    files = os.listdir(logdir)

    # 加载配置文件cfg
    cfg = None
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as f:
                cfg = json.load(f)
                pprint(cfg)

    assert cfg is not None

    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 测试集
    trainset, valset, *testset = get_dataset(cfg)
    if len(testset) == 0:
        testset = valset
    else:
        testset = testset[0]

    # 标签color map, 没有使用默认
    cmap = testset.cmap if hasattr(testset, 'cmap') else None

    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])

    # model
    model = get_model(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(logdir, use_pth)))

    # metrics
    running_metrics_val = runningScore(cfg['n_classes'], ignore_label=testset.id_background)
    time_meter = averageMeter()

    # 输出尺寸变化 && 预测图保存路径
    print(f'output size = {(cfg["image_h"], cfg["image_w"]) if target_size is None else target_size}')
    save_path = os.path.join(logdir, 'predicts')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)

    # predict and save
    with torch.no_grad():
        model.eval()

        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            time_start = time.time()
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            predict = model(image)

            predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
            label = label.squeeze(1).cpu().numpy()  # [1, 1, h, w] -> [1, h, w]
            running_metrics_val.update(label, predict)

            time_meter.update(time.time() - time_start, n=image.size(0))

            if save_predict:
                predict = predict.squeeze(0)  # [1, h, w] -> [h, w]
                predict = class_to_RGB(predict, N=cfg['n_classes'], cmap=cmap)  # 如果数据集没有给定cmap,使用默认cmap
                predict = Image.fromarray(predict)
                if target_size is not None:
                    predict = t.Resize(target_size)(predict)
                predict.save(os.path.join(save_path, sample['label_path'][0]))

    metrics = running_metrics_val.get_scores()
    for k, v in metrics[0].items():
        print(k, v)
    for k, v in metrics[1].items():
        print(k, v)
    print('inference time per image: ', time_meter.avg)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument("-d", type=str, help="predict dataset")
    parser.add_argument("-i", type=str, help="predict id")
    parser.add_argument("-s", type=bool, default=True, help="save predict or not")

    args = parser.parse_args()

    # 指定database和训练id, 即根据 eg. run/camvid/2019-10-08-11-02预测
    args.d = 'camvid'
    args.i = '2019-10-08-11-02'

    predict(args.d, args.i, save_predict=args.s)
