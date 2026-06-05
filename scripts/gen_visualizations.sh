#!/bin/bash
# Generate MeshLab-compatible PLY visualizations for all Step 2 results
set -e

PYTHON=/home/chenhui.yang/.local/share/conda/envs/any3dis/bin/python3

echo "Generating visualizations for all scenes..."
$PYTHON << 'PYEOF'
import torch, numpy as np, open3d as o3d
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

RES_DIR = Path("outputs/version_scannetpp_proc/openvocab_results")
GT_DIR = Path("data/scannetpp_processed/Scannetpp_3D/val/groundtruth")
OUT_DIR = Path("outputs/version_scannetpp_proc/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

palette = np.array([
    [255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],
    [128,0,0],[0,128,0],[0,0,128],[128,128,0],[128,0,128],[0,128,128],
    [255,128,0],[255,0,128],[128,255,0],[0,255,128],[128,0,255],[0,128,255],
    [192,192,192],[128,128,128],[255,128,128],[128,255,128],[128,128,255],
    [255,255,128],[128,255,255],[255,128,255],[64,64,64],[192,0,0],[0,192,0],
    [0,0,192],[192,192,0],[192,0,192],[0,192,192],[255,192,128],[255,128,192],
    [192,128,255],[128,255,192],[192,255,128],[128,192,255],[255,192,192],
]*10)/255.0

def rle_decode(rle):
    s = rle["counts"]
    starts, ends = np.array(s[0:][::2],dtype=np.int32)-1, np.array(s[0:][::2],dtype=np.int32)-1+np.array(s[1:][::2],dtype=np.int32)
    mask = np.zeros(rle["length"], dtype=np.uint8)
    for lo,hi in zip(np.array(s[0:][::2],dtype=np.int32)-1, np.array(s[0:][::2],dtype=np.int32)-1+np.array(s[1:][::2],dtype=np.int32)):
        mask[lo:hi]=1
    return mask

def process_scene(scene_id):
    try:
        res_path = RES_DIR / f"{scene_id}.pth"
        gt_path = GT_DIR / f"{scene_id}.pth"
        if not res_path.exists() or not gt_path.exists():
            return (scene_id, "SKIP", "missing data")

        # Check if already done
        out_sem = OUT_DIR / f"{scene_id}_semantic.ply"
        out_orig = OUT_DIR / f"{scene_id}_original.ply"
        if out_sem.exists() and out_orig.exists():
            return (scene_id, "SKIP", "already exists")

        res = torch.load(str(res_path), map_location='cpu', weights_only=False)
        gt = torch.load(str(gt_path), map_location='cpu', weights_only=False)

        xyz = gt[0] if isinstance(gt[0], np.ndarray) else gt[0].numpy()
        orig_colors = gt[1] if isinstance(gt[1], np.ndarray) else gt[1].numpy()
        if orig_colors.max() > 1.0: orig_colors = orig_colors / 255.0

        xyz = xyz.astype(np.float64)

        # Semantic colors
        sem = orig_colors.copy() * 0.3
        cls_map = {}
        for i in range(len(res['ins'])):
            m = rle_decode(res['ins'][i])
            if m.sum() == 0: continue
            if res['name'][i] not in cls_map:
                cls_map[res['name'][i]] = palette[len(cls_map) % len(palette)]
            sem[m > 0] = cls_map[res['name'][i]]

        pcd_sem = o3d.geometry.PointCloud()
        pcd_sem.points = o3d.utility.Vector3dVector(xyz)
        pcd_sem.colors = o3d.utility.Vector3dVector(sem)
        o3d.io.write_point_cloud(str(out_sem), pcd_sem)

        # Original
        pcd_orig = o3d.geometry.PointCloud()
        pcd_orig.points = o3d.utility.Vector3dVector(xyz)
        pcd_orig.colors = o3d.utility.Vector3dVector(orig_colors)
        o3d.io.write_point_cloud(str(out_orig), pcd_orig)

        n_inst = len(res['ins'])
        n_cls = len(set(res['name']))
        return (scene_id, "OK", f"{n_inst} instances, {n_cls} classes, {len(xyz)} pts")
    except Exception as e:
        return (scene_id, "FAIL", str(e)[:100])

# Get all scene IDs
scene_ids = sorted([p.stem for p in RES_DIR.glob("*.pth")])
print(f"Found {len(scene_ids)} scenes to process")

# Process with 8 workers
ok = skip = fail = 0
with Pool(processes=8) as pool:
    results = list(tqdm(pool.imap_unordered(process_scene, scene_ids), total=len(scene_ids), desc="Visualizing"))

for sid, status, msg in sorted(results, key=lambda x: x[0]):
    if status == "OK": ok += 1
    elif status == "SKIP": skip += 1
    else:
        fail += 1
        print(f"  [FAIL] {sid}: {msg}")

print(f"\nDone! OK={ok} SKIP={skip} FAIL={fail}")
print(f"Output: {OUT_DIR}")
print(f"\nFiles per scene:")
print(f"  *_semantic.ply  - semantic segmentation (colored by class)")
print(f"  *_original.ply  - original scene (RGB colors)")

# Total disk usage
import subprocess
result = subprocess.run(["du", "-sh", str(OUT_DIR)], capture_output=True, text=True)
print(f"\nTotal size: {result.stdout.strip()}")
PYEOF
