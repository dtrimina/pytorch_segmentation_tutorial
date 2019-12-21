from .metrics import averageMeter, runningScore
from .log import get_logger
from .loss import MscCrossEntropyLoss
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB



def get_dataset(cfg):
    assert cfg['dataset'] in ['ade20k', 'cityscapes', 'sunrgbd']

    if cfg['dataset'] == 'ade20k':
        from .datasets.ade20k import ADE20K
        return ADE20K(cfg, mode='train'), ADE20K(cfg, mode='val')

    if cfg['dataset'] == 'cityscapes':
        from .datasets.cityscapes import Cityscapes
        return Cityscapes(cfg, mode='train'), Cityscapes(cfg, mode='val')

    if cfg['dataset'] == 'sunrgbd':
        from .datasets.sunrgbd import SUNRGBD
        return SUNRGBD(cfg, mode='train'), SUNRGBD(cfg, mode='test')


def get_model(cfg):
    if cfg['model_name'] == 'unet':
        from .models.unet import unet
        return unet(n_classes=cfg['n_classes'])

    if cfg['model_name'] == 'drn_c_26':
        from .models.drn_c_26 import DRNSeg_C_26
        return DRNSeg_C_26(n_classes=cfg['n_classes'])

    if cfg['model_name'] == 'enet':
        from .models.enet import ENet
        return ENet(n_classes=cfg['n_classes'])

    if cfg['model_name'] == 'linknet':
        from .models.linknet import linknet
        return linknet(n_classes=cfg['n_classes'])

    if cfg['model_name'] == 'segnet':
        from .models.segnet import segnet
        return segnet(n_classes=cfg['n_classes'])

    if cfg['model_name'] == 'densenet103':
        from .models.fcdensenet import DenseNet103
        return DenseNet103(n_classes=cfg['n_classes'])

    if cfg['model_name'] == 'deeplabv3plus_resnet50':
        from .models.deeplabv3plus.deeplabv3plus import Deeplab_v3plus
        return Deeplab_v3plus(n_classes=cfg['n_classes'], backbone='resnet50')

    if cfg['model_name'] == 'deeplabv3plus_resnet101':
        from .models.deeplabv3plus.deeplabv3plus import Deeplab_v3plus
        return Deeplab_v3plus(n_classes=cfg['n_classes'], backbone='resnet101')

    raise ValueError(f'{cfg["model_name"]} not support.')