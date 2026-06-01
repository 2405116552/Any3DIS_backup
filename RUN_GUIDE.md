# Any3DIS Inference Guide

> **Any3DIS**: Open-Vocabulary 3D Instance Segmentation with Segment-Anything-Model 2 (SAM-2)

This guide covers the complete inference pipeline — from environment setup to running the full 3-step pipeline and evaluating results.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Model Weights](#model-weights)
5. [Dataset Preparation](#dataset-preparation)
6. [Configuration](#configuration)
7. [Inference Pipeline](#inference-pipeline)
8. [Evaluation](#evaluation)
9. [Visualization](#visualization)
10. [Full Benchmark](#full-benchmark)
11. [Troubleshooting](#troubleshooting)
12. [File Structure](#file-structure)

---

## Overview

Any3DIS performs **open-vocabulary 3D instance segmentation** from RGB-D video sequences of indoor scenes. The pipeline operates in three stages:

```
RGB-D Sequence → [Step 1] → Class-Agnostic 3D Masks → [Step 2] → Open-Vocab Labels → [Step 3] → Free-Form Descriptions
```

| Stage | Script | What It Does |
|-------|--------|--------------|
| **Step 1** — 3D Mask Proposals | `scripts/mask_generator.sh` | Uses SAM-2 to generate 2D masks, lifts them to 3D via RGB-D projection, and clusters into instance proposals |
| **Step 2** — Open-Vocabulary Classification | `scripts/mask_openvocab.sh` | Uses CLIP (ViT-L/14@336px) to assign categorical labels from a fixed vocabulary to each 3D instance |
| **Step 3** — Free-Vocabulary Description (optional) | `scripts/mask_freevocab.sh` | Uses OSM (OmniScient Model) to generate natural language descriptions for each instance |
| **Prompt Mode** | `scripts/mask_prompt.sh` | Interactive 3D segmentation by clicking points on 2D frames |

**Supported datasets**: ScanNet200 (198 classes), ScanNetpp (1554 classes)

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× NVIDIA GPU with 24 GB VRAM | 1× A100 40GB or A6000 48GB |
| RAM | 32 GB | 64 GB |
| Disk (code) | ~5 MB | — |
| Disk (weights) | ~2.7 GB | — |
| Disk (data) | ~50 GB | — |
| Disk (outputs) | ~500 GB per experiment | — |

**Note:** The entire ScanNet200 validation set (312 scenes) produces ~500 GB of intermediate outputs. Use a smaller split file for quick tests.

### Software

- **Linux** (tested on Ubuntu 20.04/22.04)
- **Python 3.10+**
- **CUDA 11.8** (or higher with compatible PyTorch)
- **GCC 9+** (for compiling PointNet2 CUDA ops and SAM-2)

---

## Installation

### Step 1: Install Python Dependencies

```bash
# Create and activate a conda environment (recommended)
conda create -n any3dis python=3.10 -y
conda activate any3dis

# Install PyTorch with CUDA 11.8
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies
pip install -r requirements.txt
```

### Step 2: Install SAM-2

```bash
cd segmenter2d/segment-anything-2
pip install -e . --no-build-isolation
cd ../..
```

### Step 3: Install PointNet2 CUDA Extension

```bash
cd util3d/pointnet2
python setup.py install
cd ../..
```

### Step 4: Install OmniScient-Model (for FreeVocab / Step 3 only)

```bash
git clone https://github.com/Open3DIS/OmniScient-Model.git
# The path ./OmniScient-Model is added to PYTHONPATH automatically by mask_freevocab.sh
```

### Step 5: Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
from util2d.segment_anything_v2 import SAM_L2
print('Imports OK')
"
```

---

## Model Weights

You need to download the following model weights and place them in `./weighs/`:

### 1. SAM-2 (Hiera Large)

```bash
mkdir -p weighs
wget -O weighs/sam2_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

- **Size**: ~898 MB
- **Source**: Meta AI

### 2. CLIP ViT-L/14@336px

```bash
wget -O weighs/ViT-L-14-336px.pt \
    "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
```

- **Size**: ~934 MB
- **Source**: OpenAI

### 3. OSM (OmniScient Model) — for FreeVocab only

```bash
# The OSM checkpoint should be inside the cloned OmniScient-Model repo
# Or you can place osm_final.pt in ./weighs/
```

- **Size**: ~792 MB

### Verify Weights

```bash
ls -lh weighs/
# Expected:
#   sam2_hiera_large.pt    (~898 MB)
#   ViT-L-14-336px.pt      (~934 MB)
```

---

## Dataset Preparation

Any3DIS works with two datasets: **ScanNet200** and **ScanNetpp**. The expected data directory structure is:

```
data/
├── Scannet200/
│   ├── Scannet200_2D_5interval/val/
│   │   └── {scene_id}/
│   │       ├── color/          # RGB frames (e.g., 000000.jpg)
│   │       ├── depth/          # Depth frames (e.g., 000000.png)
│   │       ├── pose/           # Camera pose files (e.g., 000000.txt)
│   │       └── intrinsic/      # Intrinsic matrix files
│   └── Scannet200_3D/val/
│       ├── original_ply_files/ # {scene_id}.ply — raw point clouds
│       ├── groundtruth/        # {scene_id}.pth — (xyz, rgb, sem, inst) tuples
│       ├── superpoints/        # {scene_id}.pth — superpoint assignments
│       ├── dc_feat_scannet200/ # {scene_id}.pth — DC features per point
│       └── isbnet_clsagnostic_scannet200/  # {scene_id}.pth — 3D proposals (optional)
└── Scannetpp/
    └── ... (same structure)
```

### Download ScanNet200 Data

1. Download the ScanNet dataset from the [official ScanNet website](http://www.scan-net.org/)
2. Extract the 2D sequences (RGB + Depth + Pose) to `data/Scannet200/Scannet200_2D_5interval/val/`
3. Extract PLY files to `data/Scannet200/Scannet200_3D/val/original_ply_files/`

### Generate 3D Ground Truth, Superpoints, and DC Features

```bash
# This script generates groundtruth, superpoints, and synthetic DC features
# It reads from data/Scannet200/Scannet200_3D/val/original_ply_files/
# and writes to the groundtruth/, superpoints/, dc_feat_scannet200/ directories
python generate_3d_data.py
```

**Note:** The DC features generated by this script are **synthetic stand-ins** (random projection of XYZ+RGB features). For reproduction of paper results, you need the real DC features from the original Open3DIS repo.

### Split Files

The `loader3d/` directory contains text files listing which scenes to process:

| File | Scenes | Purpose |
|------|--------|---------|
| `scannetv2_val.txt` | 312 | Full ScanNet200 validation set |
| `scannetv2_val_singlescene.txt` | 1 | Quick test on a single scene |
| `scannetpp_val.txt` | ~50 | ScanNetpp validation set |
| `scannetv2_train.txt` | 1201 | ScanNet200 training set |

Edit the `split_path` in your config to change which scenes to process.

---

## Configuration

All pipeline settings are controlled via YAML config files in `configs/`. Each config has these sections:

```yaml
foundation_model:
  sam2_checkpoint: './weighs/sam2_hiera_large.pt'
  clip_checkpoint: './weighs/ViT-L-14-336px.pt'

segmenter2d:
  model: 'SAM-2'           # SAM-2, Open3DIS_SAM-2, or SAI3D_SAM-2
  clip_model: 'ViT-L/14@336px'

proposal3d:
  ncluster_fps: 512         # Number of FPS clusters
  video_factor: 100         # Video-level scaling factor
  sppweight: 0.5            # Superpoint weight in clustering
  weighted_views: 10        # Number of selected keyframes
  view_optim: 10            # View optimization level (2^k)
  neighbors: 32             # KNN neighbors

data:
  dataset_name: 'scannet200'
  split_path: './loader3d/scannetv2_val.txt'
  datapath: './data/Scannet200/Scannet200_2D_5interval/val'
  gt_pth: './data/Scannet200/Scannet200_3D/val/groundtruth'
  original_ply: './data/Scannet200/Scannet200_3D/val/original_ply_files'
  spp_path: './data/Scannet200/Scannet200_3D/val/superpoints'
  cls_agnostic_3d_proposals_path: './data/.../isbnet_clsagnostic_scannet200'
  dc_features_path: './data/Scannet200/Scannet200_3D/val/dc_feat_scannet200'
  img_dim: [640, 480]
  rgb_img_dim: [1296, 968]
  img_interval: 2
  num_classes: 198

exp:
  exp_name: "my_experiment"  # Subfolder name under save_dir
  save_dir: './outputs'      # Root output directory
  mask2d_output: 'mask_sam'
  clustering_3d_output: 'mask2d_lifted'
  clip_feature: 'clip_feature'
  openvocab_output: 'openvocab_results'
  freevocab_output: 'freevocab_results'

fp16: True
```

### Available Configs

| Config | Dataset | Segmenter | Use Case |
|--------|---------|-----------|----------|
| `scannet200.yaml` | ScanNet200 | SAM-2 | Standard pipeline |
| `scannet200_freevocab_pointwise.yaml` | ScanNet200 | SAM-2 | FreeVocab with pointwise features |
| `scannetpp.yaml` | ScanNetpp | SAM-2 | Standard pipeline |
| `scannetpp_open3dis.yaml` | ScanNetpp | Open3DIS_SAM-2 | Open3DIS reproduction |
| `scannetpp_sai3d.yaml` | ScanNetpp | SAI3D_SAM-2 | SAI3D reproduction |
| `prompting/scannet200.yaml` | ScanNet200 | SAM-2 | Interactive prompt mode |
| `prompting/scannetpp.yaml` | ScanNetpp | SAM-2 | Interactive prompt mode |

### Quick Test Configuration

For a quick test, edit your config and change:
```yaml
data:
  split_path: './loader3d/scannetv2_val_singlescene.txt'  # Process only 1 scene
```

---

## Inference Pipeline

### Step 1: Class-Agnostic 3D Instance Masks

Generates 3D mask proposals by running SAM-2 on each RGB-D video and lifting 2D masks into 3D.

```bash
# ScanNet200 (default)
bash scripts/mask_generator.sh configs/scannet200.yaml

# ScanNetpp
bash scripts/mask_generator.sh configs/scannetpp.yaml

# Specify GPU
CUDA_VISIBLE_DEVICES=1 bash scripts/mask_generator.sh configs/scannet200.yaml
```

**What it does:**
1. Loads SAM-2 (Hiera Large) and CLIP models
2. For each scene, loads the RGB-D video sequence
3. Runs SAM-2 to generate 2D masks on selected keyframes
4. Lifts 2D masks to 3D using camera poses and depth
5. Clusters 3D points into instance proposals using superpoint voting
6. Saves results as RLE-encoded masks

**Output:** `outputs/{exp_name}/mask_sam/` (2D masks) and `outputs/{exp_name}/mask2d_lifted/` (3D proposals)

**Trackable:** The script uses a tracker file (`tracker_2d.txt`) to skip already-processed scenes, enabling resumption after interruption.

### Step 2: Open-Vocabulary Classification

Assigns class labels to each 3D instance proposal using CLIP.

```bash
bash scripts/mask_openvocab.sh configs/scannet200.yaml
```

**What it does:**
1. Loads the 3D proposals from Step 1
2. Computes per-point CLIP features for each scene
3. Aggregates features per instance proposal
4. Classifies each instance by comparing with text embeddings of class names
5. Assigns the closest-matching class label

**Output:** `outputs/{exp_name}/openvocab_results/`

### Step 3: Free-Vocabulary Description (Optional)

Generates natural language descriptions for each instance using the OmniScient Model (OSM).

```bash
bash scripts/mask_freevocab.sh configs/scannet200_freevocab_pointwise.yaml
```

**Prerequisite:** Clone [OmniScient-Model](https://github.com/Open3DIS/OmniScient-Model) into `./OmniScient-Model/` (the script adds it to `PYTHONPATH`).

**Output:** `outputs/{exp_name}/freevocab_results/`

### Step 4: Interactive Prompt Mode

Segment 3D objects by clicking on 2D frames.

```bash
bash scripts/mask_prompt.sh configs/prompting/scannet200.yaml
```

### Full Pipeline (Steps 1→3)

```bash
# ScanNet200 full pipeline (Step 1 + Step 3)
bash scripts/run.sh
```

---

## Evaluation

### Class-Agnostic Evaluation

Evaluates the quality of 3D mask proposals (without labels):

```bash
bash scripts/eval_classagnostic.sh configs/scannet200.yaml 2D
```

Arguments: `--config <cfg> --type [2D|3D|2D_3D]`

- `2D`: Evaluate 2D-lifted proposals (mask2d_lifted)
- `3D`: Evaluate ISBNet 3D proposals

### Open-Vocabulary Evaluation

Evaluates classification accuracy:

```bash
bash scripts/eval_openvocab.sh
# or directly:
python evaluation/eval_openvocab.py \
    --data_path ./outputs/my_experiment/openvocab_results \
    --dataset scannet200 \
    --pcl_path ./data/Scannet200/Scannet200_3D/val/groundtruth
```

### Free-Vocabulary Evaluation

Evaluates free-form description quality using Hungarian matching with sentence-transformers:

```bash
bash scripts/eval_freevocab.sh
# or directly:
python evaluation/eval_freevocab.py \
    --data_path ./outputs/my_experiment/freevocab_results \
    --dataset scannet200 \
    --pcl_path ./data/Scannet200/Scannet200_3D/val/groundtruth
```

**Note:** FreeVocab evaluation downloads `sentence-transformers/all-mpnet-base-v2` on first run (~420 MB).

---

## Visualization

Visualize 3D instance segmentation results using PyViz3D:

```bash
# ScanNet200 results
python visualization/visualize_scannet200.py

# ScanNetpp results
python visualization/visualize_scannetpp.py
```

The visualization scripts open an interactive 3D viewer in your browser showing colored instance masks with labels.

---

## Full Benchmark

To run the complete pipeline with timing and GPU monitoring:

```bash
# Run full benchmark on GPU 0
GPU_ID=0 bash run_benchmark.sh configs/scannet200.yaml

# Results are saved in benchmark_logs/{timestamp}/
# Includes:
#   - Timing per step
#   - GPU utilization history
#   - Full console logs
```

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `weighted_views` in config (e.g., from 10 to 5)
- Process scenes individually instead of batched
- Use fp16 mode (enabled by default)
- Try a GPU with more VRAM

### "No module named 'sam2'"
```bash
# Reinstall SAM-2 from local source
cd segmenter2d/segment-anything-2 && pip install -e . && cd ../..
```

### "No module named 'pointnet2_utils'"
```bash
# Recompile PointNet2 CUDA ops
cd util3d/pointnet2 && python setup.py install && cd ../..
```

### "No module named 'OmniScientModel'"
```bash
# Clone OSM (required for FreeVocab only)
git clone https://github.com/Open3DIS/OmniScient-Model.git
```
Note: The path `./OmniScient-Model` is automatically added to `PYTHONPATH` by `scripts/mask_freevocab.sh`.

### "FileNotFoundError: *_vh_clean_2.ply"
- Your data directory structure doesn't match the expected paths
- Check that `data.original_ply` in the config points to the correct PLY directory
- Check that `data.datapath` points to the correct RGB-D sequence directory

### "No such file: scannet_pc.zip"
- The `generate_3d_data.py` script expects `ScanNet_dataset/scannet_pc.zip` for generating ground truth
- Either download this zip or obtain the `.pth` ground truth files from another source

### Tracker File Issues
- If a scene crashes, remove its entry from the tracker file (e.g., `tracker_2d.txt`) to reprocess it
- Delete the tracker file to start fresh

---

## File Structure

```
Any3DIS_inference/
├── README.md                          # Package overview
├── RUN_GUIDE.md                       # This guide
├── requirements.txt                   # Python dependencies
├── setup.sh                           # One-click setup script
├── run_benchmark.sh                   # Full pipeline benchmark
├── generate_3d_data.py               # 3D data preprocessing
├── download.py                        # Dataset download helper
├── organize_scannet.py               # ScanNet data organizer
│
├── configs/                           # YAML configuration files
│   ├── scannet200.yaml               # ScanNet200 standard
│   ├── scannet200_freevocab_pointwise.yaml
│   ├── scannet200_recovery.yaml
│   ├── scannetpp.yaml                # ScanNetpp standard
│   ├── scannetpp_open3dis.yaml
│   ├── scannetpp_sai3d.yaml
│   └── prompting/                    # Interactive prompt configs
│       ├── scannet200.yaml
│       └── scannetpp.yaml
│
├── scripts/                           # Shell run scripts
│   ├── mask_generator.sh             # Step 1: 3D mask proposals
│   ├── mask_openvocab.sh             # Step 2: Open-vocab classification
│   ├── mask_freevocab.sh             # Step 3: Free-vocab description
│   ├── mask_prompt.sh                # Interactive prompt mode
│   ├── run.sh                        # Full pipeline (Step 1 + 3)
│   ├── eval_classagnostic.sh         # Evaluate mask proposals
│   ├── eval_openvocab.sh             # Evaluate classification
│   └── eval_freevocab.sh             # Evaluate descriptions
│
├── tools/                             # Python entry points
│   ├── mask_generator.py
│   ├── mask_openvocab.py
│   ├── mask_freevocab.py
│   └── mask_prompt.py
│
├── util2d/                            # 2D utilities
│   ├── segment_anything_v2.py        # SAM-2 wrapper (main)
│   ├── open3dis_sam2.py              # Open3DIS SAM-2 variant
│   ├── sai3d_sam2.py                 # SAI3D SAM-2 variant
│   ├── prompt_segment_anything_v2.py # Prompt-based SAM-2
│   ├── openai_clip.py                # CLIP model loader
│   ├── segment_anything_hq.py        # SAM-HQ wrapper
│   ├── reproduce_openvocab_pointwise.py  # OpenVocab engine
│   ├── reproduce_freevocab_pointwise.py  # FreeVocab engine
│   ├── pointwise_openvocab.py        # Alternative engines
│   ├── pointwise_freevocab.py
│   ├── sppwise_openvocab.py
│   ├── sppwise_freevocab.py
│   ├── maskwise_freevocab.py
│   ├── direct_openvocab.py
│   ├── direct_freevocab.py
│   └── util.py                       # RLE utilities
│
├── util3d/                            # 3D utilities
│   ├── gen3d_utils.py                # 3D projection & lifting
│   ├── mapper.py                     # Coordinate mapping
│   ├── ops/                          # 3D ops (CUDA)
│   │   ├── functions.py
│   │   ├── setup.py
│   │   └── src/
│   └── pointnet2/                    # PointNet2 (CUDA extension)
│       ├── pointnet2_utils.py
│       ├── pointnet2_modules.py
│       ├── setup.py
│       └── _ext_src/                 # CUDA source files
│
├── loader3d/                          # Dataset loaders
│   ├── __init__.py
│   ├── scannet200.py                 # ScanNet200 loader + class names
│   ├── scannetpp.py                  # ScanNetpp loader + class names
│   ├── scannet_loader.py            # Base ScanNet loader
│   ├── scannetv2_val.txt            # Validation split (312 scenes)
│   ├── scannetv2_val_singlescene.txt # Single scene for testing
│   ├── scannetv2_train.txt
│   ├── scannetv2_test.txt
│   ├── scannetpp_val.txt
│   ├── scannetpp_train.txt
│   └── scannetpp_test.txt
│
├── evaluation/                        # Evaluation scripts
│   ├── eval_class_agnostic.py        # Mask proposal evaluation
│   ├── eval_openvocab.py             # Classification evaluation
│   ├── eval_freevocab.py             # Description evaluation
│   ├── scannetv2_inst_eval.py        # ScanNet instance eval (official)
│   ├── scannetv2_inst_eval_freevocab.py
│   └── instance_eval_util.py
│
├── visualization/                     # Visualization
│   ├── visualize_scannet200.py
│   ├── visualize_scannetpp.py
│   └── visualize_scannetpp_benchmark_instance.py
│
└── segmenter2d/                       # SAM-2 source code
    └── segment-anything-2/
        ├── setup.py
        ├── sam2/                     # SAM-2 Python package
        │   ├── __init__.py
        │   ├── automatic_mask_generator.py
        │   ├── build_sam.py
        │   ├── sam2_image_predictor.py
        │   ├── sam2_video_predictor.py
        │   ├── modeling/             # Model architecture
        │   ├── utils/                # Utilities
        │   └── csrc/                 # CUDA kernels
        └── sam2_configs/             # SAM-2 model configs
```

---

## Quick Start (TL;DR)

```bash
# 1. Install dependencies
conda create -n any3dis python=3.10 -y && conda activate any3dis
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd segmenter2d/segment-anything-2 && pip install -e . && cd ../..
cd util3d/pointnet2 && python setup.py install && cd ../..

# 2. Download model weights
mkdir -p weighs
wget -O weighs/sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
wget -O weighs/ViT-L-14-336px.pt "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"

# 3. Prepare data (edit config to point to your data paths)
# edit configs/scannet200.yaml → set data.* paths

# 4. Run inference
bash scripts/mask_generator.sh configs/scannet200.yaml   # Step 1
bash scripts/mask_openvocab.sh configs/scannet200.yaml   # Step 2

# 5. Evaluate
python evaluation/eval_openvocab.py \
    --data_path ./outputs/{exp_name}/openvocab_results \
    --dataset scannet200
```
