#!/bin/bash
# Recover mask_sam for missing scenes on GPU 2
export PYTHONWARNINGS="ignore"
PYTHONPATH=./segmenter2d/segment-anything-2:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=2 /home/chenhui.yang/.local/share/conda/envs/any3dis/bin/python3 tools/mask_generator.py \
    --config configs/scannet200_recovery.yaml \
    --split_path loader3d/scannetv2_val_missing_mask_sam_gpu2.txt \
    --tracker tracker_2d_recovery_gpu2.txt
