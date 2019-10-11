## simple pytorch segmentation tutorial

#### version  
- python 3.7.3
- torch 1.1.0
- torchvision 0.3.0

#### database  

- [x] [CamVid](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/CamVid) : It is a automotive dataset which contains **367** training, **101** validation, and **233** testing images. There are **eleven** different classes such as building, tree, sky, car, road, etc. while the twelfth class contains unlabeled data, which we ignore during training. The original frame resolution for this dataset is 960 × 720. We use **480 x 360** images in [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid).
- [x] [SUNRGBD](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/SUNRGBD) : The SUN RGB-D dataset has **10,335** densely annotated RGB-D images taken from 20 different scenes. It also include all images data from NYU Depth v2 dataset, and selected images data from Berkeley B3DO and SUN3D dataset. Each pixel in the RGB-D images is assigned a semantic label in one of the **37 classes** or the ‘unknown’ class. It has **5285 training/validation** instances and **5050 testing** instances. If you want to train your segmentation model with **depth** information. Please see [pytorch_segmentation_RGBD链接待补充]().



#### models




model | paper | code    
:-: | :-: | :-:  
[unet](https://blog.dtrimina.cn/Segmentation/segmentation-2/) | [paper](http://xxx.itp.ac.cn/pdf/1505.04597.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/unet.py) |  
[segnet](https://blog.dtrimina.cn/Segmentation/segmentation-2/) | [paper](http://xxx.itp.ac.cn/pdf/1511.00561.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/segnet.py) |  
[LinkNet](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | [paper](http://xxx.itp.ac.cn/pdf/1707.03718.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/linknet.py)  |  
[FC-DenseNet](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | [paper](http://xxx.itp.ac.cn/pdf/1611.09326.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/fcdensenet.py) |  
[ENet](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | [paper](http://xxx.itp.ac.cn/pdf/1606.02147v1) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/enet.py) |



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

- 数据增强: RandomResizedCrop + ColorJitter + RandomFlip + RandomGrayscale
- 归一化: ToTensor + Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- 损失函数: CrossEntropyLoss + class_weight + 忽略背景
- 120 epoch, Adam optimizer, lr=1e-4, every 40 steps decay 0.1
- 支持多gpu. (eg. "gpu_ids": "0123")

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