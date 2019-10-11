import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms.transforms as t

from toolbox.datasets.augmentations import Resize


class SUNRGBD(data.Dataset):

    def __init__(self,
                 root='./database/SUNRGBD',
                 mode='train',
                 image_size=(480, 640),
                 augmentations=None,
                 use_pt_norm=True, ):

        assert mode in ['train', 'test'], f'{mode} not support.'

        self.root = root
        self.mode = mode
        self.image_size = image_size
        self.n_classes = 38  # 包括背景
        self.id_background = 0  # 背景类别id

        # 类别平衡权重 compute in database/class_weight.py
        self.class_weight = torch.tensor([ 4.23205628 , 4.94571675 , 5.57862381 , 24.44534956, 19.90010511, 9.76202942,
                                           22.72040184, 13.61414125, 25.39246383, 23.29261424, 41.05365979, 40.27980692,
                                           37.62904508, 43.93062852, 26.9015599 , 44.38626841, 36.50666673, 41.7974345,
                                           40.35631839, 40.49076758, 49.75468725, 45.17068264, 36.14867496, 44.26064598,
                                           45.11657338, 46.7591883 , 44.88959106, 47.28874519, 49.99583172, 41.15698097,
                                           40.6678578 , 48.31438898, 48.35523311, 45.27907191, 43.24865403, 45.57560593,
                                           46.90471312, 46.6266927 ],
                                         requires_grad=False)

        # 输入数据处理流程为 augmentations + transform

        # augmentations: 表示对图像的增强操作, 其中尺寸变换Resize,随机裁剪RamdomCrop, \
        #                随机旋转RandomRotation, 随机翻转RandomVerticalFlip,RandomHorizontalFlip \
        #                等改变图像形状尺寸的操作需要使image,depth,label做相同的变换,保持像素点的对应; \
        #                但像图片亮度对比度变化ColorJitter, 随机灰度化RandomGrayscale 等只对image   \
        #                进行操作, 对depth和label不做处理. 在augmentations.py里有对应的操作供参考.
        self.augmentations = augmentations

        # transform    : 将之前 augmentations后的PIL image转化为Tensor的形式,可以进行一些归一化等操作.
        # pytorch预训练模型 RGB输入统一的处理方法
        self.use_pt_norm = use_pt_norm
        self.pt_image_mean = np.asarray([0.485, 0.456, 0.406])
        self.pt_image_std = np.asarray([0.229, 0.224, 0.225])

        # 读取训练/测试图像路径信息
        with open(os.path.join(root, f'{mode}.txt'), 'r') as f:
            self.image_depth_labels = f.readlines()

    def __len__(self):
        return len(self.image_depth_labels)

    def __getitem__(self, index):
        image_path, label_path = self.image_depth_labels[index].strip().split(',')
        image = Image.open(os.path.join(self.root, image_path))  # RGB 0~255
        label = Image.open(os.path.join(self.root, label_path))  # 1 channel 0~37

        sample = {
            'image': image,
            'label': label,
        }

        if self.augmentations is not None and self.mode == 'train':
            sample = self.augmentations(sample)
        if self.mode in ['test']:
            sample = Resize(self.image_size)(sample)
        sample = self.normalize(sample)
        sample['label_path'] = label_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    def normalize(self, sample):

        # image transform
        image = sample['image']
        image = np.asarray(image, dtype=np.float64)  # 3 channel, 0~255
        image /= 255.

        if self.use_pt_norm:
            # 使用pytorch pretrained model的transform方法
            image -= self.pt_image_mean
            image /= self.pt_image_std

        image = image.transpose((2, 0, 1))  # HW3 -> 3HW
        sample['image'] = torch.from_numpy(image).float()

        # label transform
        label = np.asarray(sample['label'], dtype=np.int)
        sample['label'] = torch.from_numpy(label).long()

        return sample

    @property
    def cmap(self):
        # cmap in RedNet
        return [(0, 0, 0),
                # 0=background
                (148, 65, 137), (255, 116, 69), (86, 156, 137),
                (202, 179, 158), (155, 99, 235), (161, 107, 108),
                (133, 160, 103), (76, 152, 126), (84, 62, 35),
                (44, 80, 130), (31, 184, 157), (101, 144, 77),
                (23, 197, 62), (141, 168, 145), (142, 151, 136),
                (115, 201, 77), (100, 216, 255), (57, 156, 36),
                (88, 108, 129), (105, 129, 112), (42, 137, 126),
                (155, 108, 249), (166, 148, 143), (81, 91, 87),
                (100, 124, 51), (73, 131, 121), (157, 210, 220),
                (134, 181, 60), (221, 223, 147), (123, 108, 131),
                (161, 66, 179), (163, 221, 160), (31, 146, 98),
                (99, 121, 30), (49, 89, 240), (116, 108, 9),
                (161, 176, 169), (80, 29, 135), (177, 105, 197),
                (139, 110, 246)]  # 38


if __name__ == '__main__':
    pass

