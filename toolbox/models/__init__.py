from toolbox.models.unet import unet
from toolbox.models.segnet import segnet
from toolbox.models.linknet import linknet
from toolbox.models.fcdensenet import DenseNet103
from toolbox.models.enet import ENet
from toolbox.models.drn_c_26 import DRNSeg_C_26


def get_model(cfg):
    assert cfg['model_name'] in ['unet']

    if cfg['model_name'] == 'unet':
        return unet(n_classes=cfg['n_classes'])

    if cfg['model_name'] == 'drn_c_26':
        return DRNSeg_C_26(n_classes=cfg['n_classes'])
