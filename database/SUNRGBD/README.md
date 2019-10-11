## SUNRGBD DATA for 37 classes segmentation

网址: http://3dvision.princeton.edu/projects/2015/SUNrgbd/

我们使用处理好的RGBD images and labels by https://github.com/ankurhanda/sunrgbd-meta-data. 下载好后将图片解压到相应的文件夹。

- SUNRGBD/sunrgbd_train_image/ -> [download](http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-train_images.tgz)

- SUNRGBD/sunrgbd_test_image/  -> [download](http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-test_images.tgz)

- SUNRGBD/sunrgbd_train_test_labels/  -> [download](https://github.com/ankurhanda/sunrgbd-meta-data/blob/master/sunrgbd_train_test_labels.tar.gz)

train.txt && test.txt 每一行数据为  
image_path,label_path

#### issue 
sunrgbd_train_test_labels中有的标签图像存在噪点,即有的类别在标签图像中只有个位数的pixels,在图像尺寸变换时,这部分类别可能会消失。使用混淆矩阵计算miou，个别噪点影响不大。