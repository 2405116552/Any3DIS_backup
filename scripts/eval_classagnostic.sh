#!/bin/bash

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH

export PYTHONPATH
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 evaluation/eval_class_agnostic.py --config ${1:-configs/scannet200.yaml} --type ${2:-2D}
    
#laion2b_s39b_b160k

