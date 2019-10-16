import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms.transforms as t

from toolbox.datasets.augmentations import Resize


class ADE20K(data.Dataset):

    def __init__(self,
                 root='./database/ADEChallengeData2016',
                 mode='train',
                 image_size=(576, 576),
                 augmentations=None,
                 use_pt_norm=True, ):

        assert mode in ['train', 'val'], f'{mode} not support.'

        self.root = root
        self.mode = mode
        self.image_size = image_size
        self.n_classes = 151  # 包括背景 0~10 + 11
        self.id_unlabel = 0  # unlabel id

        # 类别平衡权重 compute in database/class_weight.py
        self.class_weight = torch.tensor([ 9.89315307 , 6.21206109 , 8.4481421  , 9.83843305 , 12.85699327, 15.28017838,
                                           16.02900972, 17.40081088, 24.0054458 , 25.74540856, 26.92861454, 26.96741779,
                                           27.953921  , 28.56669728, 29.28711489, 32.10022964, 32.91653599, 33.08942396,
                                           33.90369488, 33.57891658, 33.70651234, 34.08093056, 37.20440641, 37.96689402,
                                           38.24989009, 38.84491461, 39.03286178, 39.87486717, 40.23245337, 41.42704539,
                                           41.48714625, 41.49310821, 41.42621449, 43.57755915, 43.76779297, 44.20802952,
                                           44.11971135, 44.90655553, 45.1864103 , 45.30854919, 45.42667487, 45.35251817,
                                           45.6517232 , 45.75318838, 45.97812494, 46.1848154 , 46.27453899, 46.36730614,
                                           46.34506963, 46.47215374, 46.42763705, 46.47078568, 46.32634225, 46.492617,
                                           46.7090084 , 46.87813692, 46.6671782 , 46.6767113 , 46.79182141, 46.95538243,
                                           46.91286116, 46.94861978, 47.00641604, 47.09364093, 47.4613085 , 47.22997533,
                                           47.27846241, 47.17522216, 47.43339866, 47.41594021, 47.52206548, 47.6969829,
                                           47.79192356, 47.65914864, 47.70732011, 48.12807162, 48.04390109, 48.21207829,
                                           48.48021931, 48.3228153 , 48.43682779, 48.44695618, 48.5112262 , 48.6206623,
                                           48.52972818, 48.64451891, 48.72335011, 48.71444257, 48.77319547, 48.73911345,
                                           48.84137131, 48.78230718, 48.79662799, 48.85194903, 48.98395472, 48.91799374,
                                           48.9720186 , 49.02364135, 49.01163038, 49.00524229, 49.10688803, 49.03684315,
                                           49.00715538, 49.03168987, 49.1690763 , 49.23746747, 49.20067058, 49.25509495,
                                           49.32268837, 49.18203064, 49.23954348, 49.24538066, 49.18472259, 49.33434897,
                                           49.32015268, 49.33852018, 49.41193933, 49.33411228, 49.34416984, 49.32776956,
                                           49.30388597, 49.54014276, 49.46768643, 49.57479482, 49.55413151, 49.65661188,
                                           49.57748642, 49.6751274 , 49.53433518, 49.64964935, 49.57258304, 49.65612109,
                                           49.68198803, 49.60677851, 49.64181062, 49.65851635, 49.64705424, 49.68888967,
                                           49.68013605, 49.74373346, 49.73659667, 49.69074993, 49.7242648 , 49.77503039,
                                           49.79430621, 49.76628836, 49.71697762, 49.8866121 , 49.97872542, 49.95039807,
                                           50.0353952 ], requires_grad=False)

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
        sample['label'] = torch.from_numpy(label).long()

        return sample

    # @property
    # def cmap(self):
    #     return []


if __name__ == '__main__':
    dataset = ADE20K(root='/home/dtrimina/Desktop/pytorch_segmentation_tutorial/database/ADEChallengeData2016', mode='val',
                         use_pt_norm=True)
    from toolbox.utils import class_to_RGB

    label = dataset[1111]['label']
    label = label.numpy()

    import matplotlib.pyplot as plt

    plt.imshow(class_to_RGB(label, N=151))
    plt.show()
