#!/usr/bin/env bash
python train.py --config configs/sunrgbd_unet.json
python train.py --config configs/sunrgbd_segnet.json
python train.py --config configs/sunrgbd_linknet.json
python train.py --config configs/sunrgbd_fcdensenet.json
python train.py --config configs/sunrgbd_enet.json