#!/usr/bin/env bash
python train.py --config configs/cytiscapes_segnet.json
python train.py --config configs/cytiscapes_linknet.json
python train.py --config configs/cytiscapes_fcdensenet103.json
python train.py --config configs/cytiscapes_enet.json
python train.py --config configs/cytiscapes_drn_c_26.json

python -m torch.distributed.launch --nproc_per_node=1 train.py --config configs/ade20k_unet.json