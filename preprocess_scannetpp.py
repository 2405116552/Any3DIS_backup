#!/usr/bin/env python3
"""
Preprocess ScanNet++ data into the format expected by Any3DIS pipeline.

Input:  data/scannetpp/data_sample/<sid>/iphone/{rgb,depth,colmap}/
        data/scannetpp/pcld_0.25/<sid>.pth

Output: data/scannetpp_processed/
        ├── Scannetpp_2D_5interval/val/<sid>/
        │   ├── color/          (00000.jpg, 00001.jpg, ...)  → symlinks
        │   ├── depth/          (00000.png, 00001.png, ...)  → symlinks
        │   ├── pose/           (00000.txt, ...)              → 4×4 matrices
        │   ├── intrinsic/      (00000.txt, ...)              → per-frame intrinsics
        │   ├── intrinsic.txt
        │   └── intrinsic_depth.txt
        └── Scannetpp_3D/val/
            ├── original_ply_files/<sid>.ply
            ├── superpoints/<sid>.pth
            ├── dc_feat_scannetpp/<sid>.pth
            └── groundtruth/<sid>.pth
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import traceback

import torch
import cv2
import open3d as o3d

BASE_DIR = Path("/mnt/cpfs02/home/chenhui.yang/Any3DIS_unofficial")
SRC_2D = BASE_DIR / "data/scannetpp/data_sample"
SRC_PCLD = BASE_DIR / "data/scannetpp/pcld_0.25"
SPLIT_FILE = BASE_DIR / "loader3d/scannetpp_val.txt"
DST_BASE = BASE_DIR / "data/scannetpp_processed"

DST_2D = DST_BASE / "Scannetpp_2D_5interval" / "val"
DST_3D = DST_BASE / "Scannetpp_3D" / "val"
DST_PLY = DST_3D / "original_ply_files"
DST_SPP = DST_3D / "superpoints"
DST_DC = DST_3D / "dc_feat_scannetpp"
DST_GT = DST_3D / "groundtruth"


def load_val_scenes():
    with open(SPLIT_FILE) as f:
        return [l.strip() for l in f if l.strip()]


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion (w,x,y,z) to 3×3 rotation matrix."""
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2*qy**2 - 2*qz**2
    R[0, 1] = 2*qx*qy - 2*qz*qw
    R[0, 2] = 2*qx*qz + 2*qy*qw
    R[1, 0] = 2*qx*qy + 2*qz*qw
    R[1, 1] = 1 - 2*qx**2 - 2*qz**2
    R[1, 2] = 2*qy*qz - 2*qx*qw
    R[2, 0] = 2*qx*qz - 2*qy*qw
    R[2, 1] = 2*qy*qz + 2*qx*qw
    R[2, 2] = 1 - 2*qx**2 - 2*qy**2
    return R


def quat_trans_to_pose_matrix(qw, qx, qy, qz, tx, ty, tz):
    """Convert COLMAP quaternion+translation to 4×4 camera-to-world pose matrix."""
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def parse_colmap_images_txt(filepath):
    """Parse COLMAP images.txt, return dict: {image_name: (qw,qx,qy,qz,tx,ty,tz,camera_id)}"""
    images = {}
    with open(filepath) as f:
        lines = f.readlines()
    # Skip header lines (start with #)
    # COLMAP images.txt: each image has 2 lines (pose + points2D).
    # The points2D lines may be empty (no observations) and get filtered out.
    # We parse only lines with exactly the pose format: ID QW QX QY QZ TX TY TZ CAM_ID NAME
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        # Pose line: exactly 10 parts (or 9 without name, but standard is 10)
        # Points2D line: variable number of (X, Y, POINT3D_ID) triplets
        # We identify pose lines by checking: parts[1] should be a quaternion component (-1 to 1)
        if len(parts) == 10:
            try:
                qw = float(parts[1])
                # Sanity: qw should be a quaternion w component (roughly -1 to 1)
                if abs(qw) <= 2.0:
                    image_id = int(parts[0])
                    qx, qy, qz = float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    camera_id = int(parts[8])
                    image_name = parts[9]
                    images[image_name] = (qw, qx, qy, qz, tx, ty, tz, camera_id)
            except (ValueError, IndexError):
                continue
    return images


def parse_colmap_cameras_txt(filepath):
    """Parse COLMAP cameras.txt, return dict: {camera_id: {model, width, height, params}}"""
    cameras = {}
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 5:
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = {'model': model, 'width': width, 'height': height, 'params': params}
    return cameras


def build_intrinsic_4x4(fx, fy, cx, cy):
    """Build 4×4 intrinsic matrix from pinhole params."""
    return np.array([
        [fx, 0.0, cx, 0.0],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def build_intrinsic_3x3(fx, fy, cx, cy):
    """Build 3×3 intrinsic matrix."""
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ])


def generate_superpoints(xyz, voxel_size=0.05):
    """Voxel-based superpoint clustering."""
    xyz_min = xyz.min(axis=0)
    vi = np.floor((xyz - xyz_min) / voxel_size).astype(np.int64)
    h = vi[:, 0] * 73856093 + vi[:, 1] * 19349663 + vi[:, 2] * 83492791
    _, spp = np.unique(h, return_inverse=True)
    return spp.astype(np.int64)


def generate_dc_features(xyz, rgb):
    """Generate DC features from point cloud geometry and color.
    rgb is expected to be in [0, 1] range.
    """
    N = xyz.shape[0]
    # rgb is already [0,1], use directly
    rgb_norm = rgb.astype(np.float32)
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
    """Process one scene. Returns (scene_id, status, message)."""
    try:
        src_2d_dir = SRC_2D / scene_id / "iphone"
        src_pcld_file = SRC_PCLD / f"{scene_id}.pth"

        if not src_2d_dir.exists():
            return (scene_id, "SKIP", "Missing 2D data")
        if not src_pcld_file.exists():
            return (scene_id, "SKIP", "Missing point cloud")

        dst_scene_dir = DST_2D / scene_id
        dst_color = dst_scene_dir / "color"
        dst_depth = dst_scene_dir / "depth"
        dst_pose = dst_scene_dir / "pose"
        dst_intrinsic_dir = dst_scene_dir / "intrinsic"

        # 1. Identify frames from rgb folder
        rgb_dir = src_2d_dir / "rgb"
        depth_dir = src_2d_dir / "depth"
        if not rgb_dir.exists() or not depth_dir.exists():
            return (scene_id, "SKIP", "Missing rgb or depth folder")

        rgb_files = sorted(os.listdir(rgb_dir))
        if not rgb_files:
            return (scene_id, "SKIP", "No images")

        # 2. Parse COLMAP
        colmap_images_file = src_2d_dir / "colmap" / "images.txt"
        colmap_cameras_file = src_2d_dir / "colmap" / "cameras.txt"

        colmap_poses = {}
        if colmap_images_file.exists():
            colmap_poses = parse_colmap_images_txt(colmap_images_file)

        colmap_cameras = {}
        if colmap_cameras_file.exists():
            colmap_cameras = parse_colmap_cameras_txt(colmap_cameras_file)

        # 3. Get camera intrinsics from COLMAP (use camera_id=1)
        if 1 in colmap_cameras:
            cam = colmap_cameras[1]
            fx, fy, cx, cy = cam['params'][0], cam['params'][1], cam['params'][2], cam['params'][3]
        else:
            # Fallback default
            fx, fy, cx, cy = 1440.49, 1440.91, 959.526, 721.437

        intrinsic_4x4 = build_intrinsic_4x4(fx, fy, cx, cy)
        intrinsic_3x3 = build_intrinsic_3x3(fx, fy, cx, cy)

        # Create directories
        dst_color.mkdir(parents=True, exist_ok=True)
        dst_depth.mkdir(parents=True, exist_ok=True)
        dst_pose.mkdir(parents=True, exist_ok=True)
        dst_intrinsic_dir.mkdir(parents=True, exist_ok=True)

        # 4. Process each frame: symlink images, write pose, write intrinsic
        for new_idx, rgb_fname in enumerate(rgb_files):
            new_id = f"{new_idx:05d}"
            # 4a. Color symlink: frame_XXXXXX.jpg → 00000.jpg
            src_rgb = rgb_dir / rgb_fname
            dst_rgb = dst_color / f"{new_id}.jpg"
            if dst_rgb.exists() or dst_rgb.is_symlink():
                dst_rgb.unlink()
            os.symlink(os.path.relpath(src_rgb, dst_color.parent), dst_rgb)

            # 4b. Depth symlink: replace .jpg with .png
            depth_fname = rgb_fname.replace('.jpg', '.png')
            src_depth = depth_dir / depth_fname
            dst_depth_file = dst_depth / f"{new_id}.png"
            if src_depth.exists():
                if dst_depth_file.exists() or dst_depth_file.is_symlink():
                    dst_depth_file.unlink()
                os.symlink(os.path.relpath(src_depth, dst_depth.parent), dst_depth_file)

            # 4c. Pose file: extract from COLMAP or write identity
            dst_pose_file = dst_pose / f"{new_id}.txt"
            if dst_pose_file.exists():
                dst_pose_file.unlink()
            if rgb_fname in colmap_poses:
                qw, qx, qy, qz, tx, ty, tz, _ = colmap_poses[rgb_fname]
                pose_mat = quat_trans_to_pose_matrix(qw, qx, qy, qz, tx, ty, tz)
            else:
                pose_mat = np.eye(4)
            np.savetxt(dst_pose_file, pose_mat)

            # 4d. Per-frame intrinsic (3×3 used by compute_mapping_torch)
            dst_intrinsic_file = dst_intrinsic_dir / f"{new_id}.txt"
            if dst_intrinsic_file.exists():
                dst_intrinsic_file.unlink()
            np.savetxt(dst_intrinsic_file, intrinsic_3x3)

        # 5. Scene-level intrinsics
        intrinsic_txt = dst_scene_dir / "intrinsic.txt"
        if intrinsic_txt.exists():
            intrinsic_txt.unlink()
        np.savetxt(intrinsic_txt, intrinsic_4x4)

        depth_intrinsic_txt = dst_scene_dir / "intrinsic_depth.txt"
        if depth_intrinsic_txt.exists():
            depth_intrinsic_txt.unlink()
        # For iPhone LiDAR: use depth camera intrinsic (lower res)
        # Since depth gets resized to RGB resolution, use RGB intrinsic
        depth_intrinsic_4x4 = build_intrinsic_4x4(
            577.870605, 577.870605, 319.5, 239.5
        )
        np.savetxt(depth_intrinsic_txt, depth_intrinsic_4x4)

        # 6. Point cloud: .pth → .ply
        dst_ply = DST_PLY / f"{scene_id}.ply"
        dst_gt = DST_GT / f"{scene_id}.pth"
        dst_spp = DST_SPP / f"{scene_id}.pth"
        dst_dc = DST_DC / f"{scene_id}.pth"

        need_pc = not dst_ply.exists() or not dst_gt.exists() or not dst_spp.exists() or not dst_dc.exists()

        if need_pc:
            DST_PLY.mkdir(parents=True, exist_ok=True)
            DST_GT.mkdir(parents=True, exist_ok=True)
            DST_SPP.mkdir(parents=True, exist_ok=True)
            DST_DC.mkdir(parents=True, exist_ok=True)

            # Load point cloud from .pth
            pc_data = torch.load(str(src_pcld_file), map_location='cpu', weights_only=False)
            # Helper: convert to numpy regardless of whether already ndarray or torch tensor
            def to_np(v, dtype):
                if isinstance(v, np.ndarray):
                    return v.astype(dtype)
                return v.numpy().astype(dtype)
            xyz = to_np(pc_data['sampled_coords'], np.float64)
            colors = to_np(pc_data['sampled_colors'], np.float64)
            sem_labels = to_np(pc_data['sampled_labels'], np.int64)
            inst_labels = to_np(pc_data['sampled_instance_labels'], np.int64)

            # 6a. Write PLY
            if dst_ply.exists():
                dst_ply.unlink()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            # Colors are already in [0, 1] range (float64). Clamp to [0,1] for safety.
            colors_clamped = np.clip(colors, 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(colors_clamped)
            o3d.io.write_point_cloud(str(dst_ply), pcd)

            # 6b. Write Groundtruth
            if dst_gt.exists():
                dst_gt.unlink()
            torch.save((xyz, colors, sem_labels, inst_labels), str(dst_gt))

            # 6c. Generate Superpoints
            if dst_spp.exists():
                dst_spp.unlink()
            spp = generate_superpoints(xyz, voxel_size=0.05)
            torch.save(spp, str(dst_spp))

            # 6d. Generate DC Features
            if dst_dc.exists():
                dst_dc.unlink()
            dc_feat = generate_dc_features(xyz, colors)
            torch.save(dc_feat, str(dst_dc))

        n_frames = len(rgb_files)
        n_poses = sum(1 for f in rgb_files if f in colmap_poses)
        return (scene_id, "OK", f"{n_frames} frames, {n_poses}/{n_frames} poses from COLMAP")

    except Exception as e:
        return (scene_id, "FAIL", f"{e}\n{traceback.format_exc()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--scene", type=str, default=None, help="Process single scene (for testing)")
    args = parser.parse_args()

    scenes = load_val_scenes()
    print(f"Total scenes in split: {len(scenes)}")

    if args.scene:
        scenes = [args.scene]
        print(f"Processing single scene: {args.scene}")

    # Create output root directories
    DST_2D.mkdir(parents=True, exist_ok=True)
    DST_PLY.mkdir(parents=True, exist_ok=True)
    DST_SPP.mkdir(parents=True, exist_ok=True)
    DST_DC.mkdir(parents=True, exist_ok=True)
    DST_GT.mkdir(parents=True, exist_ok=True)

    ok = skip = fail = 0

    if args.scene or args.workers == 1:
        # Sequential mode
        for sid in tqdm(scenes, desc="Preprocessing"):
            sid, status, msg = process_scene(sid)
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skip += 1
            else:
                fail += 1
            tqdm.write(f"  [{status}] {sid}: {msg}")
    else:
        # Parallel mode
        with Pool(processes=min(args.workers, len(scenes))) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_scene, scenes),
                total=len(scenes),
                desc="Preprocessing"
            ))
        for sid, status, msg in sorted(results, key=lambda x: x[0]):
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skip += 1
            else:
                fail += 1
                print(f"  [{status}] {sid}: {msg[:200]}")

    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"  OK:   {ok}")
    print(f"  SKIP: {skip}")
    print(f"  FAIL: {fail}")
    print(f"{'='*60}")
    print(f"\nOutput structure at: {DST_BASE}")
    print(f"  2D data: {DST_2D}")
    print(f"  3D data: {DST_3D}")


if __name__ == "__main__":
    main()
