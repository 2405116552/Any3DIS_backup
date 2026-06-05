#!/bin/bash

dataset_cfg=${1:-'configs/scannetpp.yaml'}
export PYTHONWARNINGS="ignore"
PYTHONPATH=./segmenter2d/segment-anything-2:/mnt/cpfs02/home/chenhui.yang/OmniScient-Model:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=0 /home/chenhui.yang/.local/share/conda/envs/any3dis/bin/python3 tools/mask_freevocab.py --config $dataset_cfg

#  -m pdb -c continue