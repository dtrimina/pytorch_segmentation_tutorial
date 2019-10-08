#!/usr/bin/env bash
python train.py --config configs/camvid_unet.json
python train.py --config configs/camvid_segnet.json