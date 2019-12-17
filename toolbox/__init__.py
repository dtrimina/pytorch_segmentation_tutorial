from .models import get_model
from .metrics import averageMeter, runningScore

from .log import get_logger

from .loss import MscCrossEntropyLoss
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB
from .datasets.ade20k import ADE20K
from .datasets.cityscapes import Cityscapes



def get_dataset(cfg):
    assert cfg['dataset'] in ['ade20k', 'cityscapes']

    if cfg['dataset'] == 'ade20k':
        return ADE20K(cfg, mode='train'), ADE20K(cfg, mode='val')

    if cfg['dataset'] == 'cityscapes':
        return Cityscapes(cfg, mode='train'), Cityscapes(cfg, mode='val')