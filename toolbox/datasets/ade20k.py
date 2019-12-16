import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms

from toolbox.utils import color_map
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale


class ADE20K(data.Dataset):

    def __init__(self, cfg, mode='train'):

        assert mode in ['train', 'val']

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # self.dp_to_tensor = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        # ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        self.mode = mode

        with open(os.path.join(cfg['root'], f'{mode}.txt'), 'r') as f:
            self.image_depth_labels = f.readlines()

    def __len__(self):
        return len(self.image_depth_labels)

    def __getitem__(self, index):
        image_path, label_path = self.image_depth_labels[index].strip().split(',')
        image = Image.open(os.path.join(self.root, image_path)).convert('RGB')  # RGB 0~255
        label = Image.open(os.path.join(self.root, label_path))  # 1 channel 0~37

        sample = {
            'image': image,
            'label': label,
        }

        if self.mode == 'train':  # 只对训练集增强
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()

        sample['label_path'] = label_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return color_map(N=self.n_classes)


if __name__ == '__main__':
    import json
    path = '../../configs/ade20k_unet.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)


    dataset = ADE20K(cfg, mode='train')
    from toolbox.utils import class_to_RGB
    import matplotlib.pyplot as plt

    for i in range(len(dataset)):
        sample = dataset[i]

        image = sample['image']
        label = sample['label']

        image = image.numpy()
        image = image.transpose((1, 2, 0))
        image *= np.asarray([0.229, 0.224, 0.225])
        image += np.asarray([0.485, 0.456, 0.406])

        label = label.numpy()
        label = class_to_RGB(label, N=len(dataset.cmap), cmap=dataset.cmap)

        plt.subplot('121')
        plt.imshow(image)
        plt.subplot('122')
        plt.imshow(label)

        plt.show()
