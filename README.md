## simple pytorch segmentation tutorial with apex mixed precision and distributed training

#### version  
- python 3.7.5
- torch 1.3.0
- torchvision 0.4.1

#### database  

- [x] [CamVid](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/CamVid) : It is a automotive dataset which contains **367** training, **101** validation, and **233** testing images. There are **eleven** different classes such as building, tree, sky, car, road, etc. while the twelfth class contains unlabeled data, which we ignore during training. The original frame resolution for this dataset is 960 × 720. We use **480 x 360** images in [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid).
- [x] [SUNRGBD](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/SUNRGBD) : The SUN RGB-D dataset has **10,335** densely annotated RGB-D images taken from 20 different scenes. It also include all images data from NYU Depth v2 dataset, and selected images data from Berkeley B3DO and SUN3D dataset. Each pixel in the RGB-D images is assigned a semantic label in one of the **37 classes** or the ‘unknown’ class. It has **5285 training/validation** instances and **5050 testing** instances. If you want to train your segmentation model with **depth** information.
- [x] [Cityscapes](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/Cityscapes) : This dataset consists of 5000 fineannotated images, out of which **2975** are available for training, **500** for validation, and the remaining **1525** have been selected as test set. It contains **19** classes and an unlabel class. The raw resolution is 2048*1024.
- [x] [ADE20k](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/ADEChallengeData2016): ADE20K dataset is a challenge scence parsing dataset providing **150** class dense labels, which consists of **20K/2K/3K** images for training/validation/test.

#### models

model | paper | code | params size(fp32) |  
:-: | :-: | :-: | :-:   
[unet](https://blog.dtrimina.cn/Segmentation/segmentation-2/) | [paper](http://xxx.itp.ac.cn/pdf/1505.04597.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/unet.py) | 51.14MB |  
[segnet](https://blog.dtrimina.cn/Segmentation/segmentation-2/) | [paper](http://xxx.itp.ac.cn/pdf/1511.00561.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/segnet.py) | 117MB |   
[LinkNet](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | [paper](http://xxx.itp.ac.cn/pdf/1707.03718.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/linknet.py) | 44.07MB |   
[FC-DenseNet103](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | [paper](http://xxx.itp.ac.cn/pdf/1611.09326.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/fcdensenet.py) | 35.58MB |  
[ENet](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | [paper](http://xxx.itp.ac.cn/pdf/1606.02147v1) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/enet.py) | 1.34MB |  
[DRN-C-26](https://blog.dtrimina.cn/Segmentation/segmentation-4/) | [paper](http://xxx.itp.ac.cn/pdf/1705.09914v1) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/drn_c_26.py) | 78.67MB |   


#### default training config  

- data augmentation: colorjit + randomhflip + randomscale + randomcrop  
- input image normalize: ToTensor + Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- loss function: CrossEntropyLoss + class_weight  
- 90 epoch, Adam optimizer, initial_lr=0.02, poly learning rate policy with power=0.9  

#### train and evaluate

```
# edit config.json based on configs/template.json
# prepare dataset CamVid or SUNRGBD or ...

# train
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/ade20k_unet.json

# predict
python evaluate.py --logdir [run logdir] [-s] 

# Moreover, you can add [your configs].json in run_tasks.sh
sh run_tasks.sh

```

#### reference
- https://github.com/meetshah1995/pytorch-semseg  
- https://nvidia.github.io/apex/#  
- https://github.com/nvidia/apex  