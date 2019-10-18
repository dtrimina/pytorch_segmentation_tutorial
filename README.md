## simple pytorch segmentation tutorial

#### version  
- python 3.7.3
- torch 1.1.0
- torchvision 0.3.0

#### database  

- [x] [CamVid](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/CamVid) : It is a automotive dataset which contains **367** training, **101** validation, and **233** testing images. There are **eleven** different classes such as building, tree, sky, car, road, etc. while the twelfth class contains unlabeled data, which we ignore during training. The original frame resolution for this dataset is 960 × 720. We use **480 x 360** images in [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid).
- [x] [SUNRGBD](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/SUNRGBD) : The SUN RGB-D dataset has **10,335** densely annotated RGB-D images taken from 20 different scenes. It also include all images data from NYU Depth v2 dataset, and selected images data from Berkeley B3DO and SUN3D dataset. Each pixel in the RGB-D images is assigned a semantic label in one of the **37 classes** or the ‘unknown’ class. It has **5285 training/validation** instances and **5050 testing** instances. If you want to train your segmentation model with **depth** information. Please see [pytorch_segmentation_RGBD链接待补充]().
- [x] [Cityscapes](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/Cityscapes) : This dataset consists of 5000 fineannotated images, out of which **2975** are available for training, **500** for validation, and the remaining **1525** have been selected as test set. It contains **19** classes and an unlabel class. The raw resolution is 2048*1024.
- [x] [ADE20k](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/ADEChallengeData2016): ADE20K dataset is a challenge scence parsing dataset providing **150** class dense labels, which consists of **20K/2K/3K** images for training/validation/test.

#### models

The follow result are using default training config on one TAITAN V GPU. Dataset is Cityscapes. Input size is 512*1024. The following metrics are use state_dict() with the best miou on validation set during training. No multi-scale prediction.

model | paper | code | pixel acc | class acc | miou | fwiou | params(fp32) | fps | remarks | 
:-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-:  
[unet](https://blog.dtrimina.cn/Segmentation/segmentation-2/) | [paper](http://xxx.itp.ac.cn/pdf/1505.04597.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/unet.py) | 87.1 | 69.8  | 52.9 | 78.5 | 51.14MB | 18.57 | - |  
[segnet](https://blog.dtrimina.cn/Segmentation/segmentation-2/) | [paper](http://xxx.itp.ac.cn/pdf/1511.00561.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/segnet.py) | pixel acc |   |  | fwiou | 117MB | fps | - |  
[LinkNet](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | [paper](http://xxx.itp.ac.cn/pdf/1707.03718.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/linknet.py) | pixel acc |   |  | fwiou | 44.07MB | fps | - |  
[FC-DenseNet103](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | [paper](http://xxx.itp.ac.cn/pdf/1611.09326.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/fcdensenet.py) | pixel acc |   |  | fwiou | 35.58MB | - | use (256,512) input |  
[ENet](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | [paper](http://xxx.itp.ac.cn/pdf/1606.02147v1) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/enet.py) | pixel acc |   |  | fwiou | 1.34MB | fps | - |  
[DRN-C-26](https://blog.dtrimina.cn/Segmentation/segmentation-4/) | [paper](http://xxx.itp.ac.cn/pdf/1705.09914v1) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/drn_c_26.py) | pixel acc | class acc | miou | fwiou | 78.67MB | fps | remarks |   


#### project structure 

```
-configs  
    -template.json    #模板配置文件  
-database  
    -Camvid/...  
-run
    -[dataset1]
        -[log1]/
        -[log2]/ 
-toolbox  
    -datasets/
        -augmentations.py
        -[dataset1.py]
        -[dataset2.py]
    -loss/  
    -models/    # 可是实现自己的model并在__init__.py中导入  
        -[model1.py]
        -[model2.py]
    -log.py     # 获取日志logger  
    -metrics.py # 使用混淆矩阵计算metrics,ignore_index=id_background忽略背景
    -utils.py   # 默认的color_map以及上色
-train.py       # 训练保存模型在run/目录  
-predict.py     # 依据run/[database]/[run_id]预测  
-run_tasks.sh   # 按照配置文件批量训练  
```

#### default training config  

- data augmentation: RandomResizedCrop + RandomFlip
- input image normalize: ToTensor + Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- loss function: CrossEntropyLoss + class_weight
- 90 epoch, SGD optimizer, initial_lr=0.01, poly learning rate policy with power=0.9
- support multi gpus. (eg. "gpu_ids": "0123")

#### train and predict

```
# edit config.json based on configs/template.json
# prepare dataset CamVid or SUNRGBD or ...

# train
python train.py --config=configs/[your config].json

# predict
python predict.py -d [dataset] -i [run_id] [[-s]]

# Moreover, you can add [your configs].json in run_tasks.sh
sh run_tasks.sh

```

#### reference
- https://github.com/meetshah1995/pytorch-semseg