#!/usr/bin/env python3
"""
Generate missing 3D data for ScanNet200 validation scenes using Open3D:
  data/Scannet200/Scannet200_3D/val/groundtruth/<scene>.pth
  data/Scannet200/Scannet200_3D/val/superpoints/<scene>.pth
  data/Scannet200/Scannet200_3D/val/dc_feat_scannet200/<scene>.pth
"""

import json, zipfile, numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path("/mnt/cpfs02/home/chenhui.yang/Any3DIS_unofficial")
PC_ZIP = BASE_DIR / "ScanNet_dataset" / "scannet_pc.zip"
VAL_SPLIT = BASE_DIR / "loader3d" / "scannetv2_val.txt"
PLY_DIR = BASE_DIR / "data" / "Scannet200" / "Scannet200_3D" / "val" / "original_ply_files"

DST_GT = BASE_DIR / "data" / "Scannet200" / "Scannet200_3D" / "val" / "groundtruth"
DST_SPP = BASE_DIR / "data" / "Scannet200" / "Scannet200_3D" / "val" / "superpoints"
DST_DC = BASE_DIR / "data" / "Scannet200" / "Scannet200_3D" / "val" / "dc_feat_scannet200"

# --- torch.save compatible -------------------------------------------------
def save_pth(obj, path):
    import torch
    torch.save(obj, path)

# --- label mapping ---------------------------------------------------------
# Maps raw ScanNet label -> ScanNet200 class ID
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
    # Aliases from raw ScanNet
    "sofa": 5, "otherfurniture": 134, "television": 28, "computer": 54,
    "dryer": 92, "screen": 168, "railing": 79, "kitchen cabinets": 25,
    "bathroom cabinets": 182, "ceiling lamp": 177, "countertop": 30,
    "bathroom countertop": 171, "floor mat": 109, "recycle bin": 81,
    "trashbin": 178, "kitchencounter": 117, "bathroomcounter": 171,
}

def map_label(raw: str) -> int:
    return SCANNET200_LABELS.get(raw.strip().lower(), 0)


# --- superpoints -----------------------------------------------------------
def generate_superpoints(xyz, voxel_size=0.05):
    """Voxel-based superpoint clustering (fast, no merging)."""
    xyz_min = xyz.min(axis=0)
    vi = np.floor((xyz - xyz_min) / voxel_size).astype(np.int64)
    h = vi[:, 0] * 73856093 + vi[:, 1] * 19349663 + vi[:, 2] * 83492791
    _, spp = np.unique(h, return_inverse=True)
    return spp.astype(np.int64)


# --- dc_features (synthetic stand-in) --------------------------------------
def generate_dc_features(xyz, rgb_norm):
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


# --- scene processing ------------------------------------------------------
def process(scene_id, zf):
    gt_path = DST_GT / f"{scene_id}.pth"
    spp_path = DST_SPP / f"{scene_id}.pth"
    dc_path = DST_DC / f"{scene_id}.pth"
    all_exist = gt_path.exists() and spp_path.exists() and dc_path.exists()
    if all_exist:
        return "skip"

    # Load PLY with Open3D
    ply_file = PLY_DIR / f"{scene_id}.ply"
    if not ply_file.exists():
        return "fail"

    pcd = o3d.io.read_point_cloud(str(ply_file))
    xyz = np.asarray(pcd.points).astype(np.float64)
    colors = np.asarray(pcd.colors).astype(np.float64)
    N = xyz.shape[0]

    # Load segs.json from PC zip
    segs_zip = f"ScanNet_PC/{scene_id}/{scene_id}_vh_clean_2.0.010000.segs.json"
    try:
        segs = json.loads(zf.read(segs_zip))
        seg_idx = np.array(segs["segIndices"], dtype=np.int64)
    except Exception:
        seg_idx = np.zeros(N, dtype=np.int64)

    max_seg = max(seg_idx.max() + 1, 1)

    # Load aggregation.json
    agg_zip = f"ScanNet_PC/{scene_id}/{scene_id}.aggregation.json"
    try:
        agg = json.loads(zf.read(agg_zip))
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
    inst = seg2inst[si] if len(seg_idx) > 0 else np.zeros(N, dtype=np.int64)
    sem = seg2label[si] if len(seg_idx) > 0 else np.zeros(N, dtype=np.int64)
    inst[inst == -1] = 0

    # Save
    msgs = []
    if not gt_path.exists():
        DST_GT.mkdir(parents=True, exist_ok=True)
        save_pth((xyz, colors, sem, inst), str(gt_path))
        msgs.append("gt")
    if not spp_path.exists():
        DST_SPP.mkdir(parents=True, exist_ok=True)
        save_pth(generate_superpoints(xyz), str(spp_path))
        msgs.append("spp")
    if not dc_path.exists():
        DST_DC.mkdir(parents=True, exist_ok=True)
        save_pth(generate_dc_features(xyz, colors), str(dc_path))
        msgs.append("dc")

    return "+".join(msgs) if msgs else "skip"


def main():
    with open(VAL_SPLIT) as f:
        val = [l.strip() for l in f if l.strip()]
    print(f"Validation scenes: {len(val)}")

    # Which scenes have PLY files on disk
    existing_ply = set(p.stem for p in PLY_DIR.glob("*.ply"))
    valid = [s for s in val if s in existing_ply]
    print(f"Scenes with PLY on disk: {len(valid)}")

    ex_gt = set(p.stem for p in DST_GT.glob("*.pth")) if DST_GT.exists() else set()
    ex_sp = set(p.stem for p in DST_SPP.glob("*.pth")) if DST_SPP.exists() else set()
    ex_dc = set(p.stem for p in DST_DC.glob("*.pth")) if DST_DC.exists() else set()
    print(f"Existing: GT={len(ex_gt)} SPP={len(ex_sp)} DC={len(ex_dc)}")

    # Only process scenes that need at least one file
    todo = [(s, s not in ex_gt, s not in ex_sp, s not in ex_dc) for s in valid]
    todo_any = [(s, ng, ns, nd) for s, ng, ns, nd in todo if ng or ns or nd]
    print(f"Need: GT={sum(1 for _,ng,_,_ in todo_any if ng)}, "
          f"SPP={sum(1 for _,_,ns,_ in todo_any if ns)}, "
          f"DC={sum(1 for _,_,_,nd in todo_any if nd)}, "
          f"total scenes={len(todo_any)}")

    if not todo_any:
        print("All data already generated!")
        return

    # Remove old bad data (check if scene0011_00 GT has valid xyz)
    # Delete files with nans (from old buggy parser)
    import torch
    bad_gt = []
    for p in sorted(DST_GT.glob("*.pth")):
        if p.stem == "scene0011_00":
            continue
        gt = torch.load(p)
        xyz = gt[0].numpy() if hasattr(gt[0], 'numpy') else gt[0]
        if np.any(~np.isfinite(xyz)):
            bad_gt.append(p.stem)
    if bad_gt:
        print(f"Removing {len(bad_gt)} bad GT files (NaN xyz)...")
        for sid in bad_gt:
            (DST_GT / f"{sid}.pth").unlink(missing_ok=True)
            (DST_SPP / f"{sid}.pth").unlink(missing_ok=True)
            (DST_DC / f"{sid}.pth").unlink(missing_ok=True)

    # Refresh todo
    ex_gt = set(p.stem for p in DST_GT.glob("*.pth")) if DST_GT.exists() else set()
    ex_sp = set(p.stem for p in DST_SPP.glob("*.pth")) if DST_SPP.exists() else set()
    ex_dc = set(p.stem for p in DST_DC.glob("*.pth")) if DST_DC.exists() else set()
    todo_any = [(s, s not in ex_gt, s not in ex_sp, s not in ex_dc) for s in valid]
    todo_scenes = [s for s, ng, ns, nd in todo_any if ng or ns or nd]
    print(f"Scenes to process after cleanup: {len(todo_scenes)}")

    # Index PC zip for segs/aggregation
    print("Indexing PC zip...")
    with zipfile.ZipFile(PC_ZIP, "r") as zf:
        pc_scenes = set()
        for info in zf.infolist():
            if info.is_dir() and info.filename.startswith("ScanNet_PC/"):
                parts = info.filename.split("/")
                if len(parts) == 3 and parts[1]:
                    pc_scenes.add(parts[1])

    ok = fail = skip = 0
    with zipfile.ZipFile(PC_ZIP, "r") as zf:
        for sid in tqdm(todo_scenes, desc="Generating 3D data"):
            if sid not in pc_scenes:
                fail += 1
                continue
            try:
                result = process(sid, zf)
                if result == "fail":
                    fail += 1
                elif result == "skip":
                    skip += 1
                else:
                    ok += 1
            except Exception as e:
                tqdm.write(f"  [FAIL] {sid}: {e}")
                fail += 1

    print(f"\nDone! OK={ok} Skip={skip} Fail={fail}")
    print(f"  groundtruth: {len(list(DST_GT.glob('*.pth')))} files")
    print(f"  superpoints: {len(list(DST_SPP.glob('*.pth')))} files")
    print(f"  dc_features: {len(list(DST_DC.glob('*.pth')))} files")


if __name__ == "__main__":
    main()
