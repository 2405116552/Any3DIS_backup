#!/bin/bash

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH

export PYTHONPATH
CUDA_VISIBLE_DEVICES=0 /home/chenhui.yang/.local/share/conda/envs/any3dis/bin/python3 evaluation/eval_class_agnostic.py --config configs/scannet200.yaml --type 2D
    
#laion2b_s39b_b160k

