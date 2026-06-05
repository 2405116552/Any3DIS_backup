#!/bin/bash
# Full pipeline: Step1 → Step2 → Step3 on GPUs 1,4,5
set -e

export PYTHONWARNINGS="ignore"
export PYTHONPATH=./segmenter2d/segment-anything-2:./util2d:$PYTHONPATH
PYTHON=/home/chenhui.yang/.local/share/conda/envs/any3dis/bin/python3
CONFIG=configs/scannetpp_processed.yaml
LOG=./pipeline_logs/20260602_222828

run_parallel() {
    local step=$1; local tool=$2; local outdir=$3
    local extra_path=$4; local extra_args=$5
    echo "=== $step on GPUs 1,4,5 ==="
    echo "Start: $(date)"
    mkdir -p "outputs/version_scannetpp_proc/$outdir"

    # Split remaining scenes
    comm -23 <(cat loader3d/scannetpp_val.txt|sort) \
             <(ls outputs/version_scannetpp_proc/$outdir/ 2>/dev/null|sed 's/\.pth$//'|sort) \
             > /tmp/pipeline_remaining.txt
    local remaining=$(wc -l < /tmp/pipeline_remaining.txt)
    if [ "$remaining" -eq 0 ]; then
        echo "$step already complete (50/50)"
        return 0
    fi
    echo "Remaining: $remaining scenes"

    local arr=($(cat /tmp/pipeline_remaining.txt))
    local total=${#arr[@]}
    local chunk=$(( (total + 2) / 3 ))

    echo "${arr[@]:0:$chunk}" | tr ' ' '\n' > /tmp/p_gpu1.txt
    echo "${arr[@]:$chunk:$chunk}" | tr ' ' '\n' > /tmp/p_gpu4.txt
    echo "${arr[@]:$((chunk*2)):$chunk}" | tr ' ' '\n' > /tmp/p_gpu5.txt

    # Launch on 3 GPUs
    local TRACKER_BASE="/tmp/tracker_${outdir}"
    for gpu in 1 4 5; do
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$extra_path:$PYTHONPATH \
            nohup $PYTHON $tool --config $CONFIG --split_path /tmp/p_gpu${gpu}.txt \
            --tracker ${TRACKER_BASE}_gpu${gpu}.txt \
            >> "$LOG/${step}_gpu${gpu}.log" 2>&1 &
        echo "  GPU$gpu PID=$!"
    done

    # Wait for all to finish
    while true; do
        local n=$(ls outputs/version_scannetpp_proc/$outdir/*.pth 2>/dev/null|wc -l)
        local running=$(ps aux|grep "$tool"|grep python3|grep -v grep|wc -l)
        echo "[$(date +%H:%M:%S)] $step: $n/50 done, $running processes"
        if [ "$n" -ge 50 ] || [ "$running" -eq 0 ]; then break; fi
        sleep 120
    done
    echo "$step complete: $(ls outputs/version_scannetpp_proc/$outdir/*.pth 2>/dev/null|wc -l)/50 at $(date)"
}

# Step 1 is already running
echo "=== Step 1 already launched, monitoring... ==="
while true; do
    n=$(ls outputs/version_scannetpp_proc/mask2d_lifted/*.pth 2>/dev/null|wc -l)
    running=$(ps aux|grep "mask_generator"|grep python3|grep -v grep|wc -l)
    echo "[$(date +%H:%M:%S)] Step1: $n/50, $running processes"
    if [ "$n" -ge 50 ] || [ "$running" -eq 0 ]; then break; fi
    sleep 120
done
echo "Step 1 done: $(ls outputs/version_scannetpp_proc/mask2d_lifted/*.pth 2>/dev/null|wc -l)/50"

# Step 2
run_parallel "step2" "tools/mask_openvocab.py" "openvocab_results" "" ""

# Step 3
run_parallel "step3" "tools/mask_freevocab.py" "freevocab_results" \
    "/mnt/cpfs02/home/chenhui.yang/OmniScient-Model" ""

# Generate visualizations
echo "=== Generating Visualizations ==="
$PYTHON << 'PYEOF'
import torch, numpy as np, open3d as o3d
from pathlib import Path; from multiprocessing import Pool
from tqdm import tqdm; from collections import Counter

RES_DIR=Path("outputs/version_scannetpp_proc/openvocab_results")
GT_DIR=Path("data/scannetpp_processed/Scannetpp_3D/val/groundtruth")
OUT_DIR=Path("outputs/version_scannetpp_proc/visualizations")
OUT_DIR.mkdir(parents=True,exist_ok=True)

palette=np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,128,0],[255,0,128],[128,255,0],[0,255,128],[128,0,255],[0,128,255],[192,0,0],[0,192,0],[0,0,192],[192,192,0],[192,0,192],[0,192,192],[128,0,0],[0,128,0],[0,0,128],[128,128,0],[128,0,128],[0,128,128],[255,64,64],[64,255,64],[64,64,255],[255,255,64],[255,64,255],[64,255,255]]*10)/255.0

def rle_decode(rle):
    s=rle["counts"]
    starts,ends=np.array(s[0:][::2],dtype=np.int32)-1,np.array(s[0:][::2],dtype=np.int32)-1+np.array(s[1:][::2],dtype=np.int32)
    mask=np.zeros(rle["length"],dtype=np.uint8)
    for lo,hi in zip(starts,ends):mask[lo:hi]=1
    return mask

def process(scene_id):
    try:
        rp=RES_DIR/f"{scene_id}.pth"; gp=GT_DIR/f"{scene_id}.pth"; op=OUT_DIR/f"{scene_id}_seg.ply"
        if not rp.exists() or not gp.exists() or op.exists(): return (scene_id,"SKIP")
        res=torch.load(str(rp),map_location='cpu',weights_only=False)
        gt=torch.load(str(gp),map_location='cpu',weights_only=False)
        xyz=gt[0].numpy() if hasattr(gt[0],'numpy') else gt[0]
        rgb=gt[1].numpy() if hasattr(gt[1],'numpy') else gt[1]
        if rgb.max()>1:rgb/=255.0
        out=rgb.copy();cls_map={}
        for i in range(len(res['ins'])):
            m=rle_decode(res['ins'][i])
            if m.sum()==0:continue
            cn=res['name'][i]
            if cn not in cls_map:cls_map[cn]=palette[len(cls_map)%len(palette)]
            out[m>0]=cls_map[cn]*0.7+rgb[m>0]*0.3
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors=o3d.utility.Vector3dVector(out.astype(np.float64))
        o3d.io.write_point_cloud(str(op),pcd)
        return (scene_id,"OK",f"{len(res['ins'])} inst, {len(set(res['name']))} cls")
    except Exception as e: return (scene_id,"FAIL",str(e)[:100])

ids=sorted([p.stem for p in RES_DIR.glob("*.pth")])
print(f"Visualizing {len(ids)} scenes...")
with Pool(8) as pool:
    for sid,st,msg in tqdm(pool.imap_unordered(process,ids),total=len(ids)):
        if st=="FAIL":print(f"  FAIL {sid}: {msg}")

import subprocess as sp
sz=sp.run(["du","-sh",str(OUT_DIR)],capture_output=True,text=True).stdout.strip()
print(f"Done! {sz}")
PYEOF

echo ""
echo "============================================"
echo "FULL PIPELINE COMPLETE!"
echo "Step1: $(ls outputs/version_scannetpp_proc/mask2d_lifted/*.pth 2>/dev/null|wc -l)/50"
echo "Step2: $(ls outputs/version_scannetpp_proc/openvocab_results/*.pth 2>/dev/null|wc -l)/50"
echo "Step3: $(ls outputs/version_scannetpp_proc/freevocab_results/*.pth 2>/dev/null|wc -l)/50"
echo "Visualizations: $(ls outputs/version_scannetpp_proc/visualizations/*_seg.ply 2>/dev/null|wc -l)/50"
echo "Time: $(date)"
echo "============================================"
