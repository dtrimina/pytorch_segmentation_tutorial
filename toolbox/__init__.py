from .models import get_model
from .metrics import averageMeter, runningScore

from .log import get_logger

from .loss import MscCrossEntropyLoss
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB
from .datasets.ade20k import ADE20K



def get_dataset(cfg):
    assert cfg['dataset'] in ['ade20k']

    if cfg['dataset'] == 'ade20k':
        return ADE20K(cfg, mode='train'), ADE20K(cfg, mode='val')