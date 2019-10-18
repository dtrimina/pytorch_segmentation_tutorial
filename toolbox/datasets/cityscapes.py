import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms.transforms as t

from toolbox.datasets.augmentations import Resize


class Cityscapes(data.Dataset):

    def __init__(self,
                 root='./database/Cityscapes',
                 mode='train',
                 image_size=(1024, 2048),
                 augmentations=None,
                 use_pt_norm=True, ):

        assert mode in ['train', 'val'], f'{mode} not support.'

        self.root = root
        self.mode = mode
        self.image_size = image_size
        self.n_classes = 20  # 包括背景 0~10 + 11
        self.id_unlabel = 19  # unlabel id , 为了标签连续,将255变成19

        # 类别平衡权重 compute in database/class_weight.py
        self.class_weight = torch.tensor([ 3.36258308, 14.0332588, 4.9894646, 39.25158044, 36.5057383, 32.8996697,
                                           46.27562166, 40.67150678, 6.70475007, 33.55271029, 18.51486878, 32.99530091,
                                           47.68301907, 12.69611686, 45.20456246, 45.78191067, 45.82529538, 48.40734227,
                                           42.75923387, 7.88855432, ], requires_grad=False)

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
        label = Image.open(os.path.join(self.root, label_path))  # 1 channel 0~11

        sample = {
            'image': image,
            'label': label,
        }

        if self.augmentations is not None and self.mode == 'train':
            sample = self.augmentations(sample)
        if self.mode in ['val', 'test']:
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
        label[label == 255] = 19  # 为了标签连续,将255改为19
        sample['label'] = torch.from_numpy(label).long()

        return sample

    # @property
    # def cmap(self):
    #     return []


if __name__ == '__main__':
    dataset = Cityscapes(root='/home/dtrimina/Desktop/pytorch_segmentation_tutorial/database/Cityscapes',
                         use_pt_norm=True)
    from toolbox.utils import class_to_RGB

    label = dataset[456]['label']
    label = label.numpy()

    import matplotlib.pyplot as plt

    plt.imshow(class_to_RGB(label, N=20))
    plt.show()
