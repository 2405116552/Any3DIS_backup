#!/bin/bash
#
# Full benchmark pipeline for Any3DIS on ScanNet200 validation set (312 scenes)
# Records timing, logs output, monitors GPU usage
#
set -e

export PYTHONWARNINGS="ignore"
export PYTHONPATH=./segmenter2d/segment-anything-2:./util2d:$PYTHONPATH

# Use GPU 0 by default, override with GPU_ID env var
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

CONFIG=${1:-configs/scannet200.yaml}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./benchmark_logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Any3DIS Benchmark - ScanNet200 Validation (312 scenes)"
echo "GPU: $GPU_ID"
echo "Config: $CONFIG"
echo "Log: $LOG_DIR"
echo "Start: $(date)"
echo "=============================================="

# Helper: run step with timing
run_step() {
    local step_name=$1
    local cmd=$2
    local log_file="$LOG_DIR/${step_name}.log"
    local time_file="$LOG_DIR/${step_name}.time"

    echo ""
    echo "=== [$step_name] Starting at $(date) ==="
    /usr/bin/time -f "wall_clock=%E\nuser_time=%U\nsys_time=%S\nmax_mem_KB=%M" \
        -o "$time_file" \
        bash -c "$cmd" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    echo "=== [$step_name] Finished at $(date), exit=$exit_code ==="
    if [ $exit_code -ne 0 ]; then
        echo "[ERROR] Step $step_name failed with exit code $exit_code"
        return $exit_code
    fi
    return 0
}

# Start GPU monitoring in background
echo "Starting GPU monitor..."
(
    echo "timestamp,gpu_util,mem_used,mem_total,power_w,temp_c"
    while true; do
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null | \
        while IFS=, read -r idx util mem_used mem_total power temp; do
            echo "$(date +%H:%M:%S),$util,$mem_used,$mem_total,$power,$temp"
        done
        sleep 30
    done
) > "$LOG_DIR/gpu_monitor.csv" 2>/dev/null &
GPU_MONITOR_PID=$!

# Cleanup on exit
cleanup() {
    echo "Stopping GPU monitor (PID: $GPU_MONITOR_PID)"
    kill $GPU_MONITOR_PID 2>/dev/null || true
    echo "Benchmark finished at $(date)"
    echo "Logs saved to: $LOG_DIR"
}
trap cleanup EXIT

# ============================================================
# Step 1: Class-agnostic 3D instance segmentation
# ============================================================
run_step "01_mask_generator" \
    "python3 tools/mask_generator.py --config $CONFIG" || exit 1

# ============================================================
# Step 2: Open-vocabulary classification
# ============================================================
run_step "02_mask_openvocab" \
    "python3 tools/mask_openvocab.py --config $CONFIG" || exit 1

# ============================================================
# Step 3: Free-form vocabulary
# ============================================================
run_step "03_mask_freevocab" \
    "python3 tools/mask_freevocab.py --config $CONFIG" || exit 1

# ============================================================
# Evaluation
# ============================================================
echo ""
echo "=== Running Evaluation ==="

# Class-agnostic evaluation
run_step "04_eval_classagnostic" \
    "python3 evaluation/eval_class_agnostic.py --config $CONFIG --type 2D" || true

# Open-vocabulary evaluation
run_step "05_eval_openvocab" \
    "python3 evaluation/eval_openvocab.py --data_path ./outputs/\$(python3 -c 'import yaml; print(yaml.safe_load(open(\"$CONFIG\"))[\"exp\"][\"exp_name\"])')/openvocab_results" || true

# Free-vocabulary evaluation
run_step "06_eval_freevocab" \
    "python3 evaluation/eval_freevocab.py --data_path ./outputs/\$(python3 -c 'import yaml; print(yaml.safe_load(open(\"$CONFIG\"))[\"exp\"][\"exp_name\"])')/freevocab_results" || true

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "All logs: $LOG_DIR"
echo "=============================================="
