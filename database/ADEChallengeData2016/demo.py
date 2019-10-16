
# 1-20210 train
# trains = []
# for i in range(1, 20211):
#     train_image = f'images/training/ADE_train_{i:08d}.jpg'
#     train_label = f'annotations/training/ADE_train_{i:08d}.png'
#     trains.append(f'{train_image},{train_label}\n')
#
# with open('train.txt', 'w') as fp:
#     fp.writelines(trains)

# # 1-2000
# vals = []
# for i in range(1, 2001):
#     val_image = f'images/validation/ADE_val_{i:08d}.jpg'
#     val_label = f'annotations/validation/ADE_val_{i:08d}.png'
#     vals.append(f'{val_image},{val_label}\n')
#
# with open('val.txt', 'w') as fp:
#     fp.writelines(vals)

import os
from PIL import Image
import numpy as np

num_label = set()

root = 'annotations/training'
labels = os.listdir(root)
for label in labels:
    label = Image.open(os.path.join('annotations/training', label))
    label = np.asarray(label).reshape(-1)
    num_label.update(set(label.tolist()))
    print(len(num_label), num_label)