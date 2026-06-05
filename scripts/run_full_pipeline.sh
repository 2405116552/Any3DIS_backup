#!/bin/bash
#
# Full 50-scene Any3DIS inference pipeline for ScanNet++
# GPUs: 1, 3, 4
#
set -e

export PYTHONWARNINGS="ignore"
export PYTHONPATH=./segmenter2d/segment-anything-2:./util2d:$PYTHONPATH

PYTHON=/home/chenhui.yang/.local/share/conda/envs/any3dis/bin/python3
CONFIG=configs/scannetpp_processed.yaml
SPLIT_FILE=loader3d/scannetpp_val.txt
LOG_DIR="./pipeline_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Any3DIS Full Pipeline - ScanNet++ (50 scenes)"
echo "GPUs: 1, 3, 4"
echo "Config: $CONFIG"
echo "Log: $LOG_DIR"
echo "Start: $(date)"
echo "=============================================="

# Step 1: Split scenes across GPUs
echo ""
echo "=== Splitting 50 scenes across 3 GPUs ==="
SCENES=($(cat $SPLIT_FILE))
TOTAL=${#SCENES[@]}
CHUNK=$(( (TOTAL + 2) / 3 ))

# GPU 1: scenes 0-16
echo "${SCENES[@]:0:$CHUNK}" | tr ' ' '\n' > /tmp/scenes_gpu1.txt
# GPU 3: scenes 17-33
echo "${SCENES[@]:$CHUNK:$CHUNK}" | tr ' ' '\n' > /tmp/scenes_gpu3.txt
# GPU 4: scenes 34-49
echo "${SCENES[@]:$((CHUNK*2)):$CHUNK}" | tr ' ' '\n' > /tmp/scenes_gpu4.txt

for gpu in 1 3 4; do
    n=$(wc -l < /tmp/scenes_gpu${gpu}.txt)
    echo "  GPU $gpu: $n scenes"
done

# ============================================================
# Step 1: Class-agnostic 3D instance segmentation (3 GPUs parallel)
# ============================================================
echo ""
echo "=============================================="
echo "STEP 1/3: Class-agnostic 3D Instance Segmentation"
echo "Start: $(date)"
echo "=============================================="

rm -f /tmp/tracker_2d_gpu*.txt

run_step1() {
    local GPU=$1
    local SPLIT=$2
    local TRACKER=$3
    local LOG="$LOG_DIR/step1_gpu${GPU}.log"
    echo "  [GPU $GPU] Starting with $(wc -l < $SPLIT) scenes at $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON tools/mask_generator.py \
        --config $CONFIG \
        --split_path $SPLIT \
        --tracker $TRACKER \
        >> "$LOG" 2>&1
    echo "  [GPU $GPU] Finished at $(date), exit=$?"
}

run_step1 1 /tmp/scenes_gpu1.txt /tmp/tracker_2d_gpu1.txt &
PID1=$!
run_step1 3 /tmp/scenes_gpu3.txt /tmp/tracker_2d_gpu3.txt &
PID3=$!
run_step1 4 /tmp/scenes_gpu4.txt /tmp/tracker_2d_gpu4.txt &
PID4=$!

echo "Step 1 running on GPUs 1 (PID=$PID1), 3 (PID=$PID3), 4 (PID=$PID4)"
echo "Logs: $LOG_DIR/step1_gpu*.log"

# Wait for all GPU processes
wait $PID1; E1=$?
wait $PID3; E3=$?
wait $PID4; E4=$?

echo ""
echo "Step 1 complete at $(date)"
echo "  GPU 1 exit: $E1"
echo "  GPU 3 exit: $E3"
echo "  GPU 4 exit: $E4"

if [ $E1 -ne 0 ] || [ $E3 -ne 0 ] || [ $E4 -ne 0 ]; then
    echo "[WARNING] Some Step 1 processes had non-zero exit codes, check logs"
fi

# ============================================================
# Step 2: Open-vocabulary classification (GPU 1)
# ============================================================
echo ""
echo "=============================================="
echo "STEP 2/3: Open-vocabulary Classification"
echo "Start: $(date)"
echo "=============================================="

CUDA_VISIBLE_DEVICES=1 $PYTHON tools/mask_openvocab.py \
    --config $CONFIG \
    >> "$LOG_DIR/step2.log" 2>&1
E2=$?
echo "Step 2 complete at $(date), exit=$E2"

# ============================================================
# Step 3: Free-form vocabulary (GPU 1)
# ============================================================
echo ""
echo "=============================================="
echo "STEP 3/3: Free-form Vocabulary"
echo "Start: $(date)"
echo "=============================================="

rm -f /tmp/tracker_freevocab_full.txt

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./segmenter2d/segment-anything-2:/mnt/cpfs02/home/chenhui.yang/OmniScient-Model:./util2d:$PYTHONPATH \
    $PYTHON tools/mask_freevocab.py \
    --config $CONFIG \
    --tracker /tmp/tracker_freevocab_full.txt \
    >> "$LOG_DIR/step3.log" 2>&1
E3=$?
echo "Step 3 complete at $(date), exit=$E3"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=============================================="
echo "PIPELINE COMPLETE!"
echo "Finished: $(date)"
echo "Logs: $LOG_DIR"
echo "=============================================="

# Quick summary of outputs
echo ""
echo "Output summary:"
echo "  Step 1 (mask2d_lifted): $(ls outputs/version_scannetpp_proc/mask2d_lifted/*.pth 2>/dev/null | wc -l) scenes"
echo "  Step 2 (openvocab):     $(ls outputs/version_scannetpp_proc/openvocab_results/*.pth 2>/dev/null | wc -l) scenes"
echo "  Step 3 (freevocab):     $(ls outputs/version_scannetpp_proc/freevocab_results/*.pth 2>/dev/null | wc -l) scenes"
