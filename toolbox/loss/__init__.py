import torch.nn as nn
from toolbox.loss.loss import CrossEntropyLoss2D


def get_loss(cfg):

    assert cfg['loss'] in ['crossentropyloss2D']

    return {
        'crossentropyloss2D': CrossEntropyLoss2D,

    }[cfg['loss']]()
