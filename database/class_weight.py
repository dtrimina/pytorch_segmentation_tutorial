import os
from PIL import Image
import numpy as np


def linknet_class_weight(num_classes):
    p_class = num_classes / num_classes.sum()
    return 1 / (np.log(1.02 + p_class))


def count_classes(root, n_classes):
    num_classes = np.zeros(n_classes)
    for image in os.listdir(root):
        image = Image.open(os.path.join(root, image))
        image = np.asarray(image).reshape(-1)
        num = np.bincount(image)
        num_classes += num

    return num_classes


if __name__ == '__main__':
    # # compute camvid weight
    # root = 'CamVid/trainannot'
    # n_classes = 12
    # weight = linknet_class_weight(count_classes(root, n_classes))
    # print(weight)

    # # compute sunrgbd weight
    # root = 'SUNRGBD/sunrgbd_train_test_labels'
    # n_classes = 38
    # num_classes = np.zeros(n_classes)
    # for image in [f'img-{i:06d}.png' for i in range(5051, 10336)]:
    #     image = Image.open(os.path.join(root, image))
    #     image = np.asarray(image).reshape(-1)
    #     num = np.bincount(image, minlength=n_classes)
    #     num_classes += num
    # weight = linknet_class_weight(num_classes)
    # print(weight)

    # # compute cityscapes weight
    # root = 'Cityscapes'
    # n_classes = 20
    # num_classes = np.zeros(n_classes)
    # with open('Cityscapes/train.txt', 'r') as fp:
    #     pathes = fp.readlines()
    # data = set()
    # for path in pathes:
    #     _, label_path = path.strip().split(',')
    #     image = Image.open(os.path.join(root, label_path))
    #     image = np.asarray(image).reshape(-1).copy()
    #     # 为了标签连续,将255转为19,在dataset中使用同样的做法
    #     image[image == 255] = 19
    #     num = np.bincount(image, minlength=n_classes)
    #     num_classes += num
    # weight = linknet_class_weight(num_classes)
    # print(weight)

    # compute ADE20k weight
