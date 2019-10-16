## Cityscape for 19 classes segmentation
网址：https://www.cityscapes-dataset.com/

We download data from https://blog.csdn.net/zz2230633069/article/details/84591532.  
But it does not have disparity data to obtain depth, you need register and send an email in its website if you want.  

You need
```   
leftImg8bit/  
    -train/  
    -val/  
gtFine/  
    -train/  
    -val/  
```

**Cityscapes**: This dataset consists of 5000 fineannotated images, out of which **2975** are available for training, **500** for validation, and the remaining **1525** have been selected as test set. It contains **19** classes and an unlabel class. The raw resolution is 2048*1024.