import os
from PIL import Image
import numpy as np


def linknet_class_weight(num_classes):
    p_class = num_classes / num_classes.sum()
    return 1 / (np.log(1.02 + p_class))


def compute_weight(root, n_classes):
    num_classes = np.zeros(n_classes)
    for image in os.listdir(root):
        image = Image.open(os.path.join(root, image))
        image = np.asarray(image).reshape(-1)
        num = np.bincount(image)
        num_classes += num

    weight = linknet_class_weight(num_classes)
    print(weight.tolist())


root = 'CamVid/trainannot'
n_classes = 12
compute_weight(root, n_classes)

