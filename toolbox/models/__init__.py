from toolbox.models.unet import unet
from toolbox.models.segnet import segnet


def get_model(cfg):
    assert cfg['model_name'] in ['unet', 'segnet']

    return {
        'unet': unet,
        'segnet': segnet,
    }[cfg['model_name']](n_classes=cfg['n_classes'])
