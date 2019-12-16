import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale


class ADE20K(data.Dataset):

    def __init__(self, cfg, mode='train'):

        assert mode in ['train', 'val']

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg.root

        self.aug = Compose([
            ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
                saturation=cfg.saturation),
            RandomHorizontalFlip(cfg.p),
            RandomScale(cfg.scales),
            RandomCrop(cfg.crop_size, pad_if_needed=True)
        ])

        self.mode = mode

        with open(os.path.join(cfg.root, f'{mode}.txt'), 'r') as f:
            self.image_depth_labels = f.readlines()

    def __len__(self):
        return len(self.image_depth_labels)

    def __getitem__(self, index):
        image_path, depth_path, label_path = self.image_depth_labels[index].strip().split(',')
        image = Image.open(os.path.join(self.root, image_path))  # RGB 0~255
        depth = Image.open(os.path.join(self.root, depth_path)).convert('RGB')  # 1 channel -> 3
        label = Image.open(os.path.join(self.root, label_path))  # 1 channel 0~37

        sample = {
            'image': image,
            'depth': depth,
            'label': label,
        }

        if self.mode == 'train':  # 只对训练集增强
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()

        sample['label_path'] = label_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [(0, 0, 0),
                (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                (64, 0, 0), (192, 0, 0), (64, 128, 0),
                (192, 128, 0), (64, 0, 128), (192, 0, 128),
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128),
                (0, 192, 128), (128, 192, 128), (64, 64, 0), (192, 64, 0),
                (64, 192, 0), (192, 192, 0), (64, 64, 128), (192, 64, 128),
                (64, 192, 128), (192, 192, 128), (0, 0, 64), (128, 0, 64),
                (0, 128, 64), (128, 128, 64), (0, 0, 192), (128, 0, 192),
                (0, 128, 192), (128, 128, 192), (64, 0, 64)]  # 41


if __name__ == '__main__':
    class Config(object):
        def __init__(self):
            # train
            self.ims_per_gpu = 2
            self.num_workers = 4

            self.lr_start = 5e-4
            self.momentum = 0.9
            self.weight_decay = 5e-4
            self.lr_power = 0.9
            self.epochs = 300

            # model
            self.model_name = 'dual'
            self.aspp_global_feature = False

            # loss
            self.class_weight = 'no'

            # dataset
            self.name = 'nyuv2'
            self.root = '/home/dtrimina/database/NYUv2'
            self.n_classes = 41
            self.id_unlabel = 0

            # self.image_size = (480, 640)

            # augmentation
            self.brightness = 0.5
            self.contrast = 0.5
            self.saturation = 0.5
            self.p = 0.5
            # self.scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
            self.scales = (0.5, 2.0)
            self.crop_size = (480, 640)

            # eval control
            self.eval_scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
            self.eval_flip = True


    dataset = NYUv2(Config(), mode='train')
    from toolbox.utils import class_to_RGB
    import matplotlib.pyplot as plt

    for i in range(len(dataset)):
        sample = dataset[i]

        image = sample['image']
        depth = sample['depth']
        label = sample['label']

        image = image.numpy()
        image = image.transpose((1, 2, 0))
        image *= np.asarray([0.229, 0.224, 0.225])
        image += np.asarray([0.485, 0.456, 0.406])

        depth = depth.numpy()
        depth = depth.transpose((1, 2, 0))
        depth *= np.asarray([0.226, 0.226, 0.226])
        depth += np.asarray([0.449, 0.449, 0.449])

        label = label.numpy()
        label = class_to_RGB(label, N=41, cmap=dataset.cmap)

        plt.subplot('131')
        plt.imshow(image)
        plt.subplot('132')
        plt.imshow(depth)
        plt.subplot('133')
        plt.imshow(label)

        plt.show()
