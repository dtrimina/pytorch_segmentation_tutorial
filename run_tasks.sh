#!/usr/bin/env bash
python train.py --config configs/camvid_unet.json
python train.py --config configs/camvid_segnet.json
python train.py --config configs/camvid_linknet.json
python train.py --config configs/camvid_fcdensenet.json
python train.py --config configs/camvid_enet.json