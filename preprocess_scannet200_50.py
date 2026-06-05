#!/usr/bin/env python3
"""
Preprocess data/scannet200_50 for Any3DIS pipeline.
Generates: scene list, 3D data (groundtruth, superpoints, DC features),
and 2D intrinsics files.

Usage:
  python3 preprocess_scannet200_50.py
"""

import json
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import torch
import shutil

BASE_DIR = Path("/mnt/cpfs02/home/chenhui.yang/Any3DIS_unofficial")
DATA_DIR = BASE_DIR / "data" / "scannet200_50"
PC_DIR = DATA_DIR / "ScanNet_PC"
RGBD_DIR = DATA_DIR / "scannet_rgbd_sample"

# Output directories
DST_3D = DATA_DIR / "Scannet200_3D" / "val"
DST_PLY = DST_3D / "original_ply_files"
DST_GT = DST_3D / "groundtruth"
DST_SPP = DST_3D / "superpoints"
DST_DC = DST_3D / "dc_feat_scannet200"

SCENE_LIST_FILE = BASE_DIR / "loader3d" / "scannet200_50_val.txt"

# Label mapping: raw ScanNet label -> ScanNet200 class ID
SCANNET200_LABELS = {
    "wall": 0, "chair": 1, "floor": 2, "table": 3, "door": 4, "couch": 5,
    "cabinet": 6, "shelf": 7, "desk": 8, "office chair": 9, "bed": 10,
    "pillow": 11, "sink": 12, "picture": 13, "window": 14, "toilet": 15,
    "bookshelf": 16, "monitor": 17, "curtain": 18, "book": 19,
    "armchair": 20, "coffee table": 21, "box": 22, "refrigerator": 23,
    "lamp": 24, "kitchen cabinet": 25, "towel": 26, "clothes": 27,
    "tv": 28, "nightstand": 29, "counter": 30, "dresser": 31, "stool": 32,
    "cushion": 33, "plant": 34, "ceiling": 35, "bathtub": 36,
    "end table": 37, "dining table": 38, "keyboard": 39, "bag": 40,
    "backpack": 41, "toilet paper": 42, "printer": 43, "tv stand": 44,
    "whiteboard": 45, "blanket": 46, "shower curtain": 47, "trash can": 48,
    "closet": 49, "stairs": 50, "microwave": 51, "stove": 52, "shoe": 53,
    "computer tower": 54, "bottle": 55, "bin": 56, "ottoman": 57, "bench": 58,
    "board": 59, "washing machine": 60, "mirror": 61, "copier": 62,
    "basket": 63, "sofa chair": 64, "file cabinet": 65, "fan": 66,
    "laptop": 67, "shower": 68, "paper": 69, "person": 70,
    "paper towel dispenser": 71, "oven": 72, "blinds": 73, "rack": 74,
    "plate": 75, "blackboard": 76, "piano": 77, "suitcase": 78, "rail": 79,
    "radiator": 80, "recycling bin": 81, "container": 82, "wardrobe": 83,
    "soap dispenser": 84, "telephone": 85, "bucket": 86, "clock": 87,
    "stand": 88, "light": 89, "laundry basket": 90, "pipe": 91,
    "clothes dryer": 92, "guitar": 93, "toilet paper holder": 94,
    "seat": 95, "speaker": 96, "column": 97, "bicycle": 98, "ladder": 99,
    "bathroom stall": 100, "shower wall": 101, "cup": 102, "jacket": 103,
    "storage bin": 104, "coffee maker": 105, "dishwasher": 106,
    "paper towel roll": 107, "machine": 108, "mat": 109, "windowsill": 110,
    "bar": 111, "toaster": 112, "bulletin board": 113, "ironing board": 114,
    "fireplace": 115, "soap dish": 116, "kitchen counter": 117,
    "doorframe": 118, "toilet paper dispenser": 119, "mini fridge": 120,
    "fire extinguisher": 121, "ball": 122, "hat": 123,
    "shower curtain rod": 124, "water cooler": 125, "paper cutter": 126,
    "tray": 127, "shower door": 128, "pillar": 129, "ledge": 130,
    "toaster oven": 131, "mouse": 132, "toilet seat cover dispenser": 133,
    "furniture": 134, "cart": 135, "storage container": 136, "scale": 137,
    "tissue box": 138, "light switch": 139, "crate": 140,
    "power outlet": 141, "decoration": 142, "sign": 143, "projector": 144,
    "closet door": 145, "vacuum cleaner": 146, "candle": 147, "plunger": 148,
    "stuffed animal": 149, "headphones": 150, "dish rack": 151, "broom": 152,
    "guitar case": 153, "range hood": 154, "dustpan": 155, "hair dryer": 156,
    "water bottle": 157, "handicap bar": 158, "purse": 159, "vent": 160,
    "shower floor": 161, "water pitcher": 162, "mailbox": 163, "bowl": 164,
    "paper bag": 165, "alarm clock": 166, "music stand": 167,
    "projector screen": 168, "divider": 169, "laundry detergent": 170,
    "bathroom counter": 171, "object": 172, "bathroom vanity": 173,
    "closet wall": 174, "laundry hamper": 175, "bathroom stall door": 176,
    "ceiling light": 177, "trash bin": 178, "dumbbell": 179,
    "stair rail": 180, "tube": 181, "bathroom cabinet": 182,
    "cd case": 183, "closet rod": 184, "coffee kettle": 185,
    "structure": 186, "shower head": 187, "keyboard piano": 188,
    "case of water bottles": 189, "coat rack": 190, "storage organizer": 191,
    "folded chair": 192, "fire alarm": 193, "power strip": 194,
    "calendar": 195, "poster": 196, "potted plant": 197, "luggage": 198,
    "mattress": 199,
    # Aliases
    "sofa": 5, "otherfurniture": 134, "television": 28, "computer": 54,
    "dryer": 92, "screen": 168, "railing": 79, "kitchen cabinets": 25,
    "bathroom cabinets": 182, "ceiling lamp": 177, "countertop": 30,
    "bathroom countertop": 171, "floor mat": 109, "recycle bin": 81,
    "trashbin": 178, "kitchencounter": 117, "bathroomcounter": 171,
}


def map_label(raw: str) -> int:
    return SCANNET200_LABELS.get(raw.strip().lower(), 0)


def parse_scene_meta(txt_path):
    """Parse intrinsic params from scene .txt metadata file."""
    params = {}
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" = ")
            if len(parts) == 2:
                try:
                    params[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    params[parts[0].strip()] = parts[1].strip()
    return params


def generate_superpoints(xyz, voxel_size=0.05):
    """Voxel-based superpoint clustering."""
    xyz_min = xyz.min(axis=0)
    vi = np.floor((xyz - xyz_min) / voxel_size).astype(np.int64)
    h = vi[:, 0] * 73856093 + vi[:, 1] * 19349663 + vi[:, 2] * 83492791
    _, spp = np.unique(h, return_inverse=True)
    return spp.astype(np.int64)


def generate_dc_features(xyz, rgb_norm):
    """Generate synthetic DC features (32-dim)."""
    N = xyz.shape[0]
    xm = xyz.mean(axis=0, keepdims=True)
    xs = xyz.std(axis=0, keepdims=True) + 1e-6
    xn = (xyz - xm) / xs
    d = np.linalg.norm(xn, axis=1, keepdims=True)
    h = xn[:, 2:3]
    feats = np.concatenate([xn, rgb_norm, d, h, np.sin(xn), np.cos(xn)], axis=1).astype(np.float32)
    rng = np.random.RandomState(42)
    proj = rng.randn(feats.shape[1], 32).astype(np.float32) * 0.1
    dc = feats @ proj
    dc = dc / (np.linalg.norm(dc, axis=1, keepdims=True) + 1e-6)
    return dc.astype(np.float16)


def process_scene(scene_id):
    """Process a single scene: generate GT, superpoints, DC features, intrinsics."""
    pc_scene_dir = PC_DIR / scene_id
    rgbd_scene_dir = RGBD_DIR / scene_id
    ply_src = pc_scene_dir / f"{scene_id}_vh_clean_2.ply"
    meta_txt = pc_scene_dir / f"{scene_id}.txt"
    segs_json = pc_scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    agg_json = pc_scene_dir / f"{scene_id}.aggregation.json"

    results = []

    # --- 1. Symlink PLY ---
    ply_dst = DST_PLY / f"{scene_id}.ply"
    if not ply_dst.exists():
        DST_PLY.mkdir(parents=True, exist_ok=True)
        if ply_dst.is_symlink():
            ply_dst.unlink()
        # Copy instead of symlink (symlink might break with relative paths)
        shutil.copy2(str(ply_src), str(ply_dst))
        results.append("ply")

    # --- 2. Generate Groundtruth ---
    gt_dst = DST_GT / f"{scene_id}.pth"
    if not gt_dst.exists():
        pcd = o3d.io.read_point_cloud(str(ply_src))
        xyz = np.asarray(pcd.points).astype(np.float64)
        colors = np.asarray(pcd.colors).astype(np.float64)
        N = xyz.shape[0]

        # Load segs.json
        try:
            with open(segs_json) as f:
                segs = json.load(f)
            seg_idx = np.array(segs["segIndices"], dtype=np.int64)
        except Exception:
            seg_idx = np.zeros(N, dtype=np.int64)

        max_seg = max(seg_idx.max() + 1, 1)

        # Load aggregation.json
        try:
            with open(agg_json) as f:
                agg = json.load(f)
            groups = agg.get("segGroups", [])
        except Exception:
            groups = []

        seg2inst = np.full(max_seg, -1, dtype=np.int64)
        seg2label = np.zeros(max_seg, dtype=np.int64)

        for g in groups:
            iid = int(g.get("objectId", g.get("id", 0))) + 1
            lid = map_label(g.get("label", "wall"))
            for sid in g.get("segments", []):
                if 0 <= sid < max_seg:
                    seg2inst[sid] = iid
                    seg2label[sid] = lid

        si = seg_idx.clip(0, max_seg - 1)
        inst = seg2inst[si]
        sem = seg2label[si]
        inst[inst == -1] = 0

        DST_GT.mkdir(parents=True, exist_ok=True)
        torch.save((xyz, colors, sem, inst), str(gt_dst))
        results.append("gt")

    # --- 3. Generate Superpoints ---
    spp_dst = DST_SPP / f"{scene_id}.pth"
    if not spp_dst.exists():
        if 'xyz' not in dir():
            pcd = o3d.io.read_point_cloud(str(ply_src))
            xyz = np.asarray(pcd.points).astype(np.float64)
        DST_SPP.mkdir(parents=True, exist_ok=True)
        torch.save(generate_superpoints(xyz), str(spp_dst))
        results.append("spp")

    # --- 4. Generate DC Features ---
    dc_dst = DST_DC / f"{scene_id}.pth"
    if not dc_dst.exists():
        if 'xyz' not in dir():
            pcd = o3d.io.read_point_cloud(str(ply_src))
            xyz = np.asarray(pcd.points).astype(np.float64)
            colors = np.asarray(pcd.colors).astype(np.float64)
        DST_DC.mkdir(parents=True, exist_ok=True)
        torch.save(generate_dc_features(xyz, colors), str(dc_dst))
        results.append("dc")

    # --- 5. Generate intrinsic files for 2D data ---
    intrinsic_txt = rgbd_scene_dir / "intrinsic.txt"
    intrinsic_depth_txt = rgbd_scene_dir / "intrinsic_depth.txt"

    if meta_txt.exists():
        params = parse_scene_meta(meta_txt)

        # Color intrinsic (4x4)
        fx_c = params.get("fx_color", 1170.18)
        fy_c = params.get("fy_color", 1170.18)
        mx_c = params.get("mx_color", 647.75)
        my_c = params.get("my_color", 483.75)

        color_intrinsic = np.array([
            [fx_c, 0.0, mx_c, 0.0],
            [0.0, fy_c, my_c, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        np.savetxt(str(intrinsic_txt), color_intrinsic)
    else:
        # Default intrinsic
        color_intrinsic = np.array([
            [1170.18, 0.0, 647.75, 0.0],
            [0.0, 1170.18, 483.75, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        np.savetxt(str(intrinsic_txt), color_intrinsic)

    # Depth intrinsic (4x4)
    depth_intrinsic = np.array([
        [571.623718, 0.0, 319.5, 0.0],
        [0.0, 571.623718, 239.5, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    np.savetxt(str(intrinsic_depth_txt), depth_intrinsic)

    if not (rgbd_scene_dir / "intrinsic.txt").exists():
        results.append("intrinsic")

    return "+".join(results) if results else "skip"


def main():
    # Get all scenes from PC directory
    scenes = sorted([d.name for d in PC_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(scenes)} scenes in {PC_DIR}")

    # Filter: scenes that have both PC and RGBD data
    valid_scenes = []
    for s in scenes:
        if (PC_DIR / s / f"{s}_vh_clean_2.ply").exists() and (RGBD_DIR / s).exists():
            valid_scenes.append(s)
    print(f"Valid scenes (PC + RGBD): {len(valid_scenes)}")

    # Save scene list
    SCENE_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCENE_LIST_FILE, "w") as f:
        for s in valid_scenes:
            f.write(s + "\n")
    print(f"Saved scene list to {SCENE_LIST_FILE}")

    # Create directories
    for d in [DST_PLY, DST_GT, DST_SPP, DST_DC]:
        d.mkdir(parents=True, exist_ok=True)

    # Process each scene
    ok = fail = skip = 0
    for scene_id in tqdm(valid_scenes, desc="Processing scenes"):
        try:
            result = process_scene(scene_id)
            if result == "fail":
                fail += 1
                tqdm.write(f"  [FAIL] {scene_id}")
            elif result == "skip":
                skip += 1
            else:
                ok += 1
                tqdm.write(f"  [{result}] {scene_id}")
        except Exception as e:
            tqdm.write(f"  [ERROR] {scene_id}: {e}")
            fail += 1

    print(f"\nDone! OK={ok} Skip={skip} Fail={fail}")
    print(f"  PLY files: {len(list(DST_PLY.glob('*.ply')))}")
    print(f"  Groundtruth: {len(list(DST_GT.glob('*.pth')))}")
    print(f"  Superpoints: {len(list(DST_SPP.glob('*.pth')))}")
    print(f"  DC Features: {len(list(DST_DC.glob('*.pth')))}")


if __name__ == "__main__":
    main()
