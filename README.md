## simple pytorch segmentation tutorial

#### version  
- python 3.7.3
- torch 1.1.0
- torchvision 0.3.0

#### database  

- [x] [CamVid](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/CamVid)

#### models

- [x] [unet](https://blog.dtrimina.cn/Segmentation/segmentation-2/) | [paper](http://xxx.itp.ac.cn/pdf/1505.04597.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/unet.py)
- [x] [segnet](https://blog.dtrimina.cn/Segmentation/segmentation-2/) | [paper](http://xxx.itp.ac.cn/pdf/1511.00561.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/segnet.py)
- [x] [LinkNet](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | [paper](http://xxx.itp.ac.cn/pdf/1707.03718.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/linknet.py)
- [x] [FC-DenseNet](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | [paper](http://xxx.itp.ac.cn/pdf/1611.09326.pdf) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/fcdensenet.py)
- [x] [ENet](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | [paper](http://xxx.itp.ac.cn/pdf/1606.02147v1) | [code](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/models/enet.py)



#### project structure 

```
-configs  
    -template.json    #模板配置文件  
-database  
    -Camvid/...  
-run
    -camvid
        -log1/
        -log2/ 
-toolbox  
    -datasets/
    -loss/  
    -models/    # 可是实现自己的model并在__init__.py中导入  
    -log.py     # 获取日志logger  
    -metrics.py  
    -utils.py   # 默认的color_map以及上色
-train.py       # 训练保存模型在run/目录  
-predict.py     # 依据run/database/id预测  
-run_tasks.sh   # 按照配置文件批量训练  
```

#### default train config  

- 预处理: [pytorch normalize](https://github.com/dtrimina/pytorch_segmentation_tutorial/blob/master/toolbox/datasets/camvid.py#L44), 无数据增强
- CrossEntropyLoss + class_weight in linknet + 忽略背景
- 120 epoch, lr=1e-4, every 40 steps decay 0.1
- Adam optimizer
- 支持多gpu使用 "gpu_ids": "0123"

#### reference
- https://github.com/meetshah1995/pytorch-semseg