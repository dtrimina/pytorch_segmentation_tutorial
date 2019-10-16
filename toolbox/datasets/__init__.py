from toolbox.datasets.camvid import CamVid
from toolbox.datasets.sunrgbd import SUNRGBD
from toolbox.datasets.cityscapes import Cityscapes
from toolbox.datasets.ade20k import ADE20K
from toolbox.datasets.augmentations import Compose, Resize, CenterCrop, RandomScale, RandomResizedCrop,\
    Lambda, RandomApply, RandomChoice, RandomOrder, RandomCrop, RandomHorizontalFlip, \
    RandomVerticalFlip, ColorJitter, RandomRotation, Grayscale, RandomGrayscale


def get_dataset(cfg):
    assert cfg['dataset'] in ['camvid', 'sunrgbd', 'cityscapes', 'ade20k']
    assert cfg['use_pt_norm'] in ['True' or 'False']

    if cfg['augmentation'] == 'default':
        augmentation = Compose([
            RandomResizedCrop(size=(cfg['image_h'], cfg['image_w']), scale=(0.85, 1), ratio=(4/5, 5/4)),
            # Resize((cfg['image_h'], cfg['image_w'])),
            # RandomScale(scale=(1, 1.2)),
            # RandomCrop(((cfg['image_h'], cfg['image_w']))),
            ColorJitter(0.05, 0.05, 0.05, 0.05),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            # RandomRotation(5),
            RandomGrayscale(0.03),
        ])
    elif cfg['augmentation'] == 'no':
        # resize 到固定尺寸
        augmentation = Resize((cfg['image_h'], cfg['image_w']))
    else:
        # augmentation = Compose([
        #     ...
        # ])
        raise('You need edit this.')

    args = {
        'image_size': (cfg['image_h'], cfg['image_w']),
        'augmentations': augmentation,
        'use_pt_norm': True if cfg['use_pt_norm'] == 'True' else False,
    }

    if cfg['dataset'] == 'camvid':
        return CamVid(mode='train', **args), CamVid(mode='val', **args), CamVid(mode='test', **args)
    elif cfg['dataset'] == 'sunrgbd':
        return SUNRGBD(mode='train', **args), SUNRGBD(mode='test', **args)
    elif cfg['dataset'] == 'cityscapes':
        return Cityscapes(mode='train', **args), Cityscapes(mode='val', **args)
    elif cfg['dataset'] == 'ade20k':
        return ADE20K(mode='train', **args), ADE20K(mode='val', **args)
    else:
        return
