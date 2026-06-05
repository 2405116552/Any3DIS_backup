#!/bin/bash
# Step 1: Class-agnostic 3D instance segmentation on 4 GPUs (0,1,4,5)
# Data: data/scannet200_50, Config: configs/scannet200_50.yaml

set -e
export PYTHONWARNINGS="ignore"
export PYTHONPATH=./segmenter2d/segment-anything-2:./util2d:$PYTHONPATH
PYTHON=/home/chenhui.yang/.local/share/conda/envs/any3dis/bin/python3
CONFIG=configs/scannet200_50.yaml
LOG_DIR=./pipeline_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="outputs/version_scannet200_50/mask2d_lifted"
TRACKER_DIR="/tmp/tracker_scannet200_50"

mkdir -p "$LOG_DIR" "$OUTDIR" "$TRACKER_DIR"
SCENE_LIST="loader3d/scannet200_50_val.txt"
TOTAL=$(wc -l < "$SCENE_LIST")
echo "Total scenes: $TOTAL"

# Get remaining scenes (not yet processed)
REMAINING_FILE="/tmp/scannet200_50_remaining_${TIMESTAMP}.txt"
comm -23 <(cat "$SCENE_LIST" | sort) \
         <(ls "$OUTDIR"/*.pth 2>/dev/null | sed 's|.*/||; s|\.pth$||' | sort) \
         > "$REMAINING_FILE"
REMAINING=$(wc -l < "$REMAINING_FILE")
echo "Remaining: $REMAINING scenes"

if [ "$REMAINING" -eq 0 ]; then
    echo "Step 1 already complete ($TOTAL/$TOTAL)"
    exit 0
fi

# Split remaining scenes into 4 groups
ARR=($(cat "$REMAINING_FILE"))
TOTAL_REM=${#ARR[@]}
CHUNK=$(( (TOTAL_REM + 3) / 4 ))

echo "${ARR[@]:0:$CHUNK}" | tr ' ' '\n' > /tmp/s1_gpu0.txt
echo "${ARR[@]:$CHUNK:$CHUNK}" | tr ' ' '\n' > /tmp/s1_gpu1.txt
echo "${ARR[@]:$((CHUNK*2)):$CHUNK}" | tr ' ' '\n' > /tmp/s1_gpu4.txt
echo "${ARR[@]:$((CHUNK*3)):$CHUNK}" | tr ' ' '\n' > /tmp/s1_gpu5.txt

for gpu in 0 1 4 5; do
    echo "GPU $gpu: $(wc -l < /tmp/s1_gpu${gpu}.txt) scenes"
done

# Launch on 4 GPUs
echo "Launching Step 1 on GPUs 0,1,4,5 at $(date)..."
for gpu in 0 1 4 5; do
    CUDA_VISIBLE_DEVICES=$gpu \
        nohup $PYTHON tools/mask_generator.py \
            --config $CONFIG \
            --split_path /tmp/s1_gpu${gpu}.txt \
            --tracker ${TRACKER_DIR}/tracker_gpu${gpu}.txt \
            >> "$LOG_DIR/step1_gpu${gpu}_${TIMESTAMP}.log" 2>&1 &
    echo "  GPU$gpu PID=$!"
done

echo ""
echo "All 4 GPUs launched. Monitoring progress..."
echo "Logs: $LOG_DIR/step1_gpu*_${TIMESTAMP}.log"
echo "Output: $OUTDIR/"
echo ""
echo "Monitor command:"
echo "  watch -n 30 'ls $OUTDIR/*.pth 2>/dev/null | wc -l'"

# Wait for completion
while true; do
    n=$(ls "$OUTDIR"/*.pth 2>/dev/null | wc -l)
    running=$(ps aux | grep "mask_generator" | grep python3 | grep -v grep | wc -l)
    echo "[$(date +%H:%M:%S)] Step1: $n/$TOTAL done, $running processes running"
    if [ "$n" -ge "$TOTAL" ] || [ "$running" -eq 0 ]; then
        break
    fi
    sleep 120
done

echo ""
echo "Step 1 complete: $(ls $OUTDIR/*.pth 2>/dev/null | wc -l)/$TOTAL at $(date)"
