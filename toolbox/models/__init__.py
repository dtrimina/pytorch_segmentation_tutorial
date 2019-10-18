from toolbox.models.unet import unet
from toolbox.models.segnet import segnet
from toolbox.models.linknet import linknet
from toolbox.models.fcdensenet import DenseNet103
from toolbox.models.enet import ENet
from toolbox.models.drn_c_26 import DRNSeg_C_26


def get_model(cfg):
    assert cfg['model_name'] in ['unet', 'segnet', 'linknet', 'fcdensenet', 'enet', 'drn']

    return {
        'unet': unet,
        'segnet': segnet,
        'linknet': linknet,
        'fcdensenet': DenseNet103,
        'enet': ENet,
        'drn': DRNSeg_C_26

    }[cfg['model_name']](n_classes=cfg['n_classes'])
