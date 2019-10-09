from toolbox.datasets.camvid import CamVid
# from toolbox.datasets.sunrgbd import SUNRGBD
from toolbox.datasets.augmentations import Compose, Resize, CenterCrop, \
    Lambda, RandomApply, RandomChoice, RandomOrder, RandomCrop, RandomHorizontalFlip, \
    RandomVerticalFlip, ColorJitter, RandomRotation, Grayscale, RandomGrayscale


def get_dataset(cfg):
    assert cfg['dataset'] in ['camvid']
    assert cfg['use_pt_norm'] in ['True' or 'False']

    if cfg['augmentation'] != 'None':
        # augmentation = Compose([
        #       ....
        # ])
        raise ('you should add augmentation here.')
    else:
        augmentation = None

    args = {
        'image_size': (cfg['image_h'], cfg['image_w']),
        'augmentations': augmentation,
        'use_pt_norm': True if cfg['use_pt_norm'] == 'True' else False,
    }

    if cfg['dataset'] == 'camvid':
        return CamVid(mode='train', **args), CamVid(mode='val', **args), CamVid(mode='test', **args)
    # elif cfg['dataset'] == 'sunrgbd':
    #     return SUNRGBD(mode='train', **args), SUNRGBD(mode='test', **args)
    else:
        return
