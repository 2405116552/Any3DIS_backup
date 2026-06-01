#!/usr/bin/env python3
"""
Organize ScanNet dataset from downloaded zip files into the structure
expected by Any3DIS.

Expected output structure:
data/Scannet200/
├── Scannet200_2D_5interval/val/<scene>/
│   ├── color/    (*.jpg, 5-digit zero-padded sequential)
│   ├── depth/    (*.png)
│   ├── pose/     (*.txt)
│   ├── intrinsic.txt
│   └── intrinsic_depth.txt
└── Scannet200_3D/val/
    └── original_ply_files/<scene>.ply
"""

import os
import sys
import zipfile
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = Path("/mnt/cpfs02/home/chenhui.yang/Any3DIS_unofficial")
PC_ZIP = BASE_DIR / "ScanNet_dataset" / "scannet_pc.zip"
RGBD_ZIP = BASE_DIR / "ScanNet_dataset" / "scannet_rgbd_sample.zip"
VAL_SPLIT = BASE_DIR / "loader3d" / "scannetv2_val.txt"

DST_2D = BASE_DIR / "data" / "Scannet200" / "Scannet200_2D_5interval" / "val"
DST_3D_PLY = BASE_DIR / "data" / "Scannet200" / "Scannet200_3D" / "val" / "original_ply_files"


def load_val_scenes():
    with open(VAL_SPLIT) as f:
        return [line.strip() for line in f if line.strip()]


def parse_intrinsic_from_txt(content: str):
    """Parse camera intrinsic from ScanNet scene metadata txt file."""
    lines = content.strip().split("\n")
    params = {}
    for line in lines:
        parts = line.strip().split(" = ")
        if len(parts) == 2:
            params[parts[0].strip()] = parts[1].strip()

    fx_c = float(params.get("fx_color", 1170.18))
    fy_c = float(params.get("fy_color", 1170.18))
    mx_c = float(params.get("mx_color", 647.75))
    my_c = float(params.get("my_color", 483.75))

    return np.array([
        [fx_c, 0.0, mx_c, 0.0],
        [0.0, fy_c, my_c, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def depth_intrinsic():
    return np.array([
        [577.870605, 0.0, 319.5, 0.0],
        [0.0, 577.870605, 239.5, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def build_index(zip_path, prefix, scene_names):
    """Build scene -> frame mapping from zip in one pass."""
    scene_frames = defaultdict(lambda: defaultdict(dict))  # scene -> frame_num -> {subdir: filename}
    scene_ply = {}  # scene -> ply_filename in zip
    scene_meta = {}  # scene -> meta txt filename in zip

    with zipfile.ZipFile(zip_path, "r") as zf:
        if prefix == "ScanNet_PC/":
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = info.filename.split("/")
                if len(parts) < 3:
                    continue
                scene = parts[1]
                fname = parts[2]

                if scene not in scene_names:
                    continue

                if fname.endswith("_vh_clean_2.ply") and not fname.endswith(".labels.ply"):
                    scene_ply[scene] = info.filename
                elif fname == f"{scene}.txt":
                    scene_meta[scene] = info.filename

        elif prefix == "scannet_rgbd_sample/":
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = info.filename.split("/")
                if len(parts) < 4:
                    continue
                scene = parts[1]
                subdir = parts[2]
                fname = parts[3]

                if scene not in scene_names:
                    continue
                if subdir not in ("color", "depth", "pose"):
                    continue

                stem = os.path.splitext(fname)[0]
                try:
                    frame_num = int(stem)
                except ValueError:
                    continue

                scene_frames[scene][frame_num][subdir] = fname

    return scene_frames, scene_ply, scene_meta


def process_scenes(scenes, pc_zf, rgbd_zf, scene_frames, scene_ply, scene_meta):
    """Process all scenes sequentially."""
    success = 0
    fail = 0
    skip = 0

    for scene_id in tqdm(scenes, desc="Organizing"):
        # --- 3D PLY ---
        ply_dst = DST_3D_PLY / f"{scene_id}.ply"
        if not ply_dst.exists() and scene_id in scene_ply:
            try:
                data = pc_zf.read(scene_ply[scene_id])
                DST_3D_PLY.mkdir(parents=True, exist_ok=True)
                with open(ply_dst, "wb") as f:
                    f.write(data)
            except Exception as e:
                tqdm.write(f"  [WARN] {scene_id}: PLY error: {e}")

        # --- Intrinsic ---
        if scene_id in scene_meta:
            txt_content = pc_zf.read(scene_meta[scene_id]).decode("utf-8")
            intrinsic = parse_intrinsic_from_txt(txt_content)
        else:
            intrinsic = parse_intrinsic_from_txt("")

        # --- 2D RGB-D ---
        scene_2d_dir = DST_2D / scene_id
        color_dir = scene_2d_dir / "color"
        depth_dir = scene_2d_dir / "depth"
        pose_dir = scene_2d_dir / "pose"

        # Skip if already has data
        if scene_id not in scene_frames or not scene_frames[scene_id]:
            existing_color = color_dir.exists() and len(list(color_dir.iterdir())) > 0 if color_dir.exists() else False
            if existing_color:
                skip += 1
                continue
            tqdm.write(f"  [WARN] {scene_id}: No RGB-D frames in zip")
            fail += 1
            continue

        # Skip if already complete
        if color_dir.exists() and len(list(color_dir.iterdir())) > 0 and \
           depth_dir.exists() and len(list(depth_dir.iterdir())) > 0 and \
           pose_dir.exists() and len(list(pose_dir.iterdir())) > 0:
            # Still ensure intrinsic exists
            intrinsic_path = scene_2d_dir / "intrinsic.txt"
            if not intrinsic_path.exists():
                np.savetxt(intrinsic_path, intrinsic)
            intrinsic_depth_path = scene_2d_dir / "intrinsic_depth.txt"
            if not intrinsic_depth_path.exists():
                np.savetxt(intrinsic_depth_path, depth_intrinsic())
            skip += 1
            continue

        frames = scene_frames[scene_id]
        sorted_frame_nums = sorted(frames.keys())

        scene_2d_dir.mkdir(parents=True, exist_ok=True)
        color_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)
        pose_dir.mkdir(exist_ok=True)

        try:
            for idx, frame_num in enumerate(sorted_frame_nums):
                new_name = f"{idx:05d}"
                fm = frames[frame_num]

                for subdir, fname in fm.items():
                    zip_path = f"scannet_rgbd_sample/{scene_id}/{subdir}/{fname}"
                    data = rgbd_zf.read(zip_path)
                    ext = os.path.splitext(fname)[1]
                    dst_dir = {"color": color_dir, "depth": depth_dir, "pose": pose_dir}[subdir]
                    with open(dst_dir / f"{new_name}{ext}", "wb") as f:
                        f.write(data)

            np.savetxt(scene_2d_dir / "intrinsic.txt", intrinsic)
            np.savetxt(scene_2d_dir / "intrinsic_depth.txt", depth_intrinsic())
            success += 1
        except Exception as e:
            tqdm.write(f"  [ERROR] {scene_id}: {e}")
            fail += 1

    return success, fail, skip


def main():
    scenes = load_val_scenes()
    val_set = set(scenes)
    print(f"Total validation scenes: {len(scenes)}")

    # Build index from both zips
    print("Indexing PC zip...")
    scene_frames_pc, scene_ply, scene_meta = build_index(str(PC_ZIP), "ScanNet_PC/", val_set)

    print("Indexing RGB-D zip...")
    scene_frames_rgbd, _, _ = build_index(str(RGBD_ZIP), "scannet_rgbd_sample/", val_set)

    pc_available = set(scene_ply.keys())
    rgbd_available = set(scene_frames_rgbd.keys())

    print(f"Scenes with PLY: {len(pc_available)}")
    print(f"Scenes with RGB-D: {len(rgbd_available)}")

    missing_pc = val_set - pc_available
    missing_rgbd = val_set - rgbd_available

    if missing_pc:
        print(f"Missing PLY: {len(missing_pc)}")
        for s in list(missing_pc)[:5]:
            print(f"  {s}")
    if missing_rgbd:
        print(f"Missing RGB-D: {len(missing_rgbd)}")
        for s in list(missing_rgbd)[:5]:
            print(f"  {s}")

    valid_scenes = [s for s in scenes if s in pc_available and s in rgbd_available]
    print(f"\nScenes available in both zips: {len(valid_scenes)}/{len(scenes)}")

    # Process
    print("\nProcessing scenes...")
    with zipfile.ZipFile(PC_ZIP, "r") as pc_zf, \
         zipfile.ZipFile(RGBD_ZIP, "r") as rgbd_zf:
        success, fail, skip = process_scenes(valid_scenes, pc_zf, rgbd_zf, scene_frames_rgbd, scene_ply, scene_meta)

    print(f"\nDone! Success: {success}, Failed: {fail}, Skipped: {skip}")

    # Remaining 3D data (groundtruth, superpoints, isbnet, dc_feat) need external tools
    total_ply = len(list(DST_3D_PLY.glob("*.ply"))) if DST_3D_PLY.exists() else 0
    total_2d = len(list(DST_2D.iterdir())) if DST_2D.exists() else 0
    print(f"\nCurrent data/Scannet200 state:")
    print(f"  2D scenes: {total_2d}")
    print(f"  PLY files: {total_ply}")


if __name__ == "__main__":
    main()
