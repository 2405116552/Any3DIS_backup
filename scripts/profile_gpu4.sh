#!/bin/bash
# GPU 4: 单场景 profiling (scene0081_00, 29帧, 快速)
export PYTHONWARNINGS="ignore"
PYTHONPATH=./segmenter2d/segment-anything-2:$PYTHONPATH
export PYTHONPATH
echo "===== PROFILING scene0081_00 ====="
START=$(date +%s)
CUDA_VISIBLE_DEVICES=4 /home/chenhui.yang/.local/share/conda/envs/any3dis/bin/python3 tools/mask_generator.py \
    --config configs/scannet200_recovery.yaml \
    --split_path loader3d/scannetv2_val_profile_gpu4.txt \
    --tracker tracker_profile_gpu4.txt
END=$(date +%s)
echo "===== TOTAL WALL TIME: $((END - START)) seconds ====="
