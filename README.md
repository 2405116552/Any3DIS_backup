# Any3DIS: Open-Vocabulary 3D Instance Segmentation with SAM-2

Unofficial implementation of Any3DIS — an open-vocabulary 3D instance segmentation system that combines **Segment Anything Model 2 (SAM-2)** for 2D mask proposals, **CLIP** for open-vocabulary classification, and **OmniScient Model (OSM)** for free-form instance descriptions.

Built on top of [Open3DIS](https://github.com/VinAIResearch/Open3DIS).

## Features

- **Class-Agnostic 3D Instance Proposals**: Lift SAM-2's 2D segmentations to 3D via RGB-D projection
- **Open-Vocabulary Classification**: Assign categorical labels from a fixed vocabulary using CLIP
- **Free-Vocabulary Descriptions**: Generate natural language descriptions using OSM
- **Interactive Prompt Mode**: Click-to-segment in 3D
- **Multi-Dataset Support**: ScanNet200 (198 classes) and ScanNetpp (1554 classes)
- **Multiple Segmenters**: SAM-2, Open3DIS SAM-2, SAI3D SAM-2

## Quick Start

```bash
# See RUN_GUIDE.md for detailed instructions

# 1. Install
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd segmenter2d/segment-anything-2 && pip install -e . && cd ../..
cd util3d/pointnet2 && python setup.py install && cd ../..

# 2. Download weights
mkdir -p weighs
wget -O weighs/sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
wget -O weighs/ViT-L-14-336px.pt "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"

# 3. Run inference
bash scripts/run.sh
```

## Pipeline

```
RGB-D Sequence
    │
    ▼
[Step 1: mask_generator.sh]  ──  SAM-2 2D masks → 3D projection → instance proposals
    │
    ▼
[Step 2: mask_openvocab.sh]  ──  CLIP feature extraction → per-instance classification
    │
    ▼
[Step 3: mask_freevocab.sh]  ──  OSM free-form description generation (optional)
    │
    ▼
3D Instances with labels/descriptions
```

## Requirements

- Linux + Python 3.10+ + CUDA 11.8+
- GPU with ≥24 GB VRAM (recommended: A100 40GB / A6000 48GB)
- ~2.7 GB model weights + ~50 GB dataset

## Documentation

- **[RUN_GUIDE.md](RUN_GUIDE.md)** — Full installation, configuration, and usage guide
- **[configs/](configs/)** — YAML configuration files for all datasets and modes
- **[scripts/](scripts/)** — Shell scripts for each pipeline step

## License

This project is built on open-source components:
- SAM-2: [Apache 2.0](https://github.com/facebookresearch/sam2)
- CLIP: [MIT](https://github.com/openai/CLIP)
- Open3DIS: [AGPL-3.0](https://github.com/VinAIResearch/Open3DIS)
