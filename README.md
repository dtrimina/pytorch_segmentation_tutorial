## simple pytorch segmentation tutorial

#### version  
- python 3.7.3
- torch 1.1.0
- torchvision 0.3.0

#### database  

- [x] [CamVid链接待补充]()

#### models

- [x] [unet](https://blog.dtrimina.cn/Segmentation/segmentation-2/) | [paper](http://xxx.itp.ac.cn/pdf/1505.04597.pdf) | [code链接待补充]()
- [x] [segnet](https://blog.dtrimina.cn/Segmentation/segmentation-2/) | [paper](http://xxx.itp.ac.cn/pdf/1511.00561.pdf) | [code链接待补充]()


#### project structure 

-configs  
&emsp;  -template.json  #模板配置文件  
-database  
&emsp;  -Camvid/...  
-run  
-toolbox  
&emsp;  -datasets/  
&emsp;  -loss/  
&emsp;  -models/  #可是实现自己的model并在__init__.py中导入  
&emsp;  -log.py  #获取日志logger  
&emsp;  -metrics.py  
&emsp;  -utils.py  
-train.py  #训练保存模型在run/目录  
-predict.py  #依据logdir预测  
-run_tasks.sh  #按照配置文件批量训练  

#### reference
- https://github.com/meetshah1995/pytorch-semseg