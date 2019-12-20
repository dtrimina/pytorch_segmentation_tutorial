import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms

from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale


class Cityscapes(data.Dataset):

    def __init__(self, cfg, mode='train'):

        assert mode in ['train', 'val'], f'{mode} not support.'

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
        # The values associated with the 35 classes
        self.full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                             32, 33, -1)
        # The values above are remapped to the following
        self.new_classes = (0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0, 7,
                            8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 17, 18, 19, 0)

        self.class_weight = np.array([9.36879574, 3.05247989, 14.15357162, 5.58607574, 37.43976198, 36.06995197,
                                      32.03698029, 46.44198507, 40.61829658, 7.04938217, 33.43519928, 20.94698721,
                                      29.2455243, 45.64950437, 11.10225935, 43.37297876, 45.06890502, 45.21967197,
                                      47.53300041, 41.05393685, ])

        with open(os.path.join(cfg['root'], f'{mode}.txt'), 'r') as f:
            self.image_depth_labels = f.readlines()

    def __len__(self):
        return len(self.image_depth_labels)

    def __getitem__(self, index):
        image_path, label_path = self.image_depth_labels[index].strip().split(',')
        image = Image.open(os.path.join(self.root, image_path)).convert('RGB')  # RGB 0~255
        label = Image.open(os.path.join(self.root, label_path))  # 1 channel 0~37
        label = self.remap(label, self.full_classes, self.new_classes)

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
        return [
            [0, 0, 0],  # unlabel
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
        ]

    def remap(self, image, old_values, new_values):
        assert isinstance(image, Image.Image) or isinstance(
            image, np.ndarray), "image must be of type PIL.Image or numpy.ndarray"
        assert type(new_values) is tuple, "new_values must be of type tuple"
        assert type(old_values) is tuple, "old_values must be of type tuple"
        assert len(new_values) == len(
            old_values), "new_values and old_values must have the same length"

        # If image is a PIL.Image convert it to a numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Replace old values by the new ones
        tmp = np.zeros_like(image)
        for old, new in zip(old_values, new_values):
            # Since tmp is already initialized as zeros we can skip new values
            # equal to 0
            if new != 0:
                tmp[image == old] = new

        return Image.fromarray(tmp)


if __name__ == '__main__':
    import json

    path = '../../configs/cityscape_drn_c_26.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)

    dataset = Cityscapes(cfg, mode='train')
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
