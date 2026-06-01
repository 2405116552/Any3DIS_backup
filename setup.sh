#!/bin/bash
#
# Any3DIS Inference Setup Script
# This script sets up the environment and downloads required model weights.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
set -e

echo "=============================================="
echo "  Any3DIS Inference Environment Setup"
echo "=============================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -------- 1. Detect Python --------
PYTHON_BIN=${PYTHON_BIN:-python3}
echo "[1/6] Checking Python: $PYTHON_BIN"
$PYTHON_BIN --version || { echo "ERROR: python3 not found. Install Python 3.10+ first."; exit 1; }

# -------- 2. Create conda/venv (optional) --------
if [[ "$USE_CONDA" == "true" ]]; then
    echo "[2/6] Creating conda environment 'any3dis'..."
    conda create -n any3dis python=3.10 -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate any3dis
    PYTHON_BIN=python
fi

# -------- 3. Install PyTorch (CUDA 11.8) --------
echo "[3/6] Installing PyTorch 2.3.1 with CUDA 11.8..."
$PYTHON_BIN -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# -------- 4. Install Python dependencies --------
echo "[4/6] Installing Python dependencies from requirements.txt..."
$PYTHON_BIN -m pip install -r requirements.txt

# -------- 5. Install SAM-2 --------
echo "[5/6] Installing Segment-Anything-2 (SAM-2)..."
cd segmenter2d/segment-anything-2
$PYTHON_BIN -m pip install -e . --no-build-isolation
cd "$SCRIPT_DIR"

# -------- 6. Install PointNet2 CUDA ops --------
echo "[6/6] Installing PointNet2 CUDA extension..."
cd util3d/pointnet2
$PYTHON_BIN setup.py install
cd "$SCRIPT_DIR"

# -------- 7. Download model weights --------
echo ""
echo "=============================================="
echo "  Model Weights Setup"
echo "=============================================="
echo ""
echo "You need to download the following model weights manually and place them in ./weighs/:"
echo ""
echo "  1. SAM-2 (sam2_hiera_large.pt) - ~900 MB"
echo "     URL: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
echo ""
echo "  2. CLIP ViT-L/14@336px (ViT-L-14-336px.pt) - ~935 MB"
echo "     URL: https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
echo ""
echo "  3. OmniScient-Model (optional, for FreeVocab) - ~800 MB"
echo "     Clone: https://github.com/Open3DIS/OmniScient-Model"
echo "     Place OSM checkpoint (osm_final.pt) in ./weighs/"
echo ""
echo "Download commands:"
echo "  mkdir -p weighs"
echo "  wget -O weighs/sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
echo "  wget -O weighs/ViT-L-14-336px.pt https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Download model weights as shown above"
echo "  2. Prepare dataset (see RUN_GUIDE.md)"
echo "  3. Run inference:  bash scripts/run.sh"
echo ""
