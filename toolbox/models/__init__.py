from toolbox.models.unet import unet
from toolbox.models.segnet import segnet
from toolbox.models.linknet import linknet
from toolbox.models.fcdensenet import DenseNet103


def get_model(cfg):
    assert cfg['model_name'] in ['unet', 'segnet', 'linknet', 'fcdensenet']

    return {
        'unet': unet,
        'segnet': segnet,
        'linknet': linknet,
        'fcdensenet': DenseNet103,

    }[cfg['model_name']](n_classes=cfg['n_classes'])
