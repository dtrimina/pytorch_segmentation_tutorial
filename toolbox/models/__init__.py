from toolbox.models.unet import unet
from toolbox.models.segnet import segnet
from toolbox.models.linknet import linknet


def get_model(cfg):
    assert cfg['model_name'] in ['unet', 'segnet', 'linknet']

    return {
        'unet': unet,
        'segnet': segnet,
        'linknet': linknet
    }[cfg['model_name']](n_classes=cfg['n_classes'])
