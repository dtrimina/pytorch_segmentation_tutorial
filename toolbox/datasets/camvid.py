import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms.transforms as t


class CamVid(data.Dataset):

    def __init__(self,
                 root='./database/CamVid',
                 mode='train',
                 image_size=(360, 480),
                 augmentations=None,
                 use_pt_norm=True, ):

        assert mode in ['train', 'val', 'test'], f'{mode} not support.'

        self.root = root
        self.mode = mode
        self.image_size = image_size
        self.n_classes = 12  # 包括背景 0~10 + 11
        self.id_background = 11  # 背景类别id

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

        if self.augmentations is not None:
            sample = self.augmentations(sample)
        sample = self.normalize(sample)
        sample['label_path'] = label_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    def normalize(self, sample):

        # image transform
        image = sample['image']
        image = t.Resize(self.image_size)(image)
        image = np.asarray(image, dtype=np.float64)  # 3 channel, 0~255
        image /= 255.

        if self.use_pt_norm:
            # 使用pytorch pretrained model的transform方法
            image -= self.pt_image_mean
            image /= self.pt_image_std

        image = image.transpose((2, 0, 1))  # HW3 -> 3HW
        sample['image'] = torch.from_numpy(image).float()

        # label transform
        label = sample['label']
        classes = np.unique(np.asarray(label))
        label = t.Resize(self.image_size, interpolation=Image.NEAREST)(label)
        label = np.asarray(label, dtype=np.int)
        assert np.all(classes == np.unique(label))  # 尺寸变换后,类别不变
        label = torch.from_numpy(label).long()
        sample['label'] = label

        return sample

    @property
    def cmap(self):
        return [[128, 128, 128],  # Sky 0
                [128, 0, 0],  # Building 1
                [192, 192, 128],  # Pole 2
                [128, 64, 128],  # Road 3
                [60, 40, 222],  # Pavement 4
                [128, 128, 0],  # Tree 5
                [192, 128, 128],  # SignSymbol 6
                [64, 64, 128],  # Fence 7
                [64, 0, 128],  # Car 8
                [64, 64, 0],  # Pedestrian 9
                [0, 128, 192],   # Bicyclist 10
                [0, 0, 0],  # Unlabelled 11
                ]


if __name__ == '__main__':
    dataset = CamVid(root='../../database/CamVid', use_pt_norm=True)

    # 上色示例
    from toolbox.utils import class_to_RGB

    for sample in dataset:
        label = sample['label']
        # RGB = class_to_RGB(label, N=dataset.n_classes, cmap=dataset.cmap)
        # im = Image.fromarray(RGB)
        # im.save(f'path/{sample["label_path"]}')
