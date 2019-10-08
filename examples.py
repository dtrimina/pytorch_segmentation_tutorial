# ########## example 1. camvid数据库真实标签上色 #########
#
# from PIL import Image
# import matplotlib.pyplot as plt
# from toolbox.datasets.camvid import CamVid
# from toolbox.utils import class_to_RGB
#
# dataset = CamVid(root='./database/CamVid', use_pt_norm=True)
#
# for sample in dataset:
#     label = sample['label']
#     RGB = class_to_RGB(label, N=dataset.n_classes, cmap=dataset.cmap)
#
#     # # show
#     # plt.imshow(RGB)
#     # plt.show()
#
#     # # save
#     # im = Image.fromarray(RGB)
#     # im.save(f'path/{sample["label_path"]}')

##########
