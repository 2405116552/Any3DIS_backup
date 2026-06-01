import os
import yaml
import torch
import argparse
import numpy as np
from munch import Munch
from tqdm import tqdm, trange

# Util
from util2d.util import masks_to_rle
from util2d.segment_anything_v2 import SAM_L2
from util2d.open3dis_sam2 import Open3DIS_SAM_L2
from util2d.sai3d_sam2 import SAI3D_SAM_L2

def rle_encode_gpu_batch(masks):
    """
    Encode RLE (Run-length-encode) from 1D binary mask.
    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    n_inst, length = masks.shape[:2]
    zeros_tensor = torch.zeros((n_inst, 1), dtype=torch.bool, device=masks.device)
    masks = torch.cat([zeros_tensor, masks, zeros_tensor], dim=1)

    rles = []
    for i in range(n_inst):
        mask = masks[i]
        runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1

        runs[1::2] -= runs[::2]

        counts = runs.cpu().numpy()
        rle = dict(length=length, counts=counts)
        rles.append(rle)
    return rles

############################################## Mask Generator ##############################################

np.random.seed(0)
torch.manual_seed(0)


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration FreeVocab")
    parser.add_argument("--config",type=str,required = True,help="Config")
    parser.add_argument("--tracker",type=str,default="tracker_2d.txt",help="Tracker file path")
    parser.add_argument("--split_path",type=str,default="",help="Override split_path in config")
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    # Scannet split path
    split_file = args.split_path if args.split_path else cfg.data.split_path
    with open(split_file, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])


    # Fondation model loader
    if cfg.segmenter2d.model == 'SAM-2':
        model = SAM_L2(cfg)
    elif cfg.segmenter2d.model == 'Open3DIS_SAM-2':
        model = Open3DIS_SAM_L2(cfg)
    elif cfg.segmenter2d.model == 'SAI3D_SAM-2':
        model = SAI3D_SAM_L2(cfg)

    # Directory Init
    save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
    mask2d_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output)

    os.makedirs(save_dir_cluster, exist_ok=True)
    os.makedirs(mask2d_path, exist_ok=True)

    # Proces every scene
    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids):
            #####################################
            # # Tracker
            done = False
            path = scene_id + ".pth"
            tracker_file = args.tracker
            if os.path.exists(tracker_file):
                with open(tracker_file, "r") as file:
                    lines = file.readlines()
                    lines = [line.strip() for line in lines]
                    for line in lines:
                        if path in line:
                            done = True
                            break
            if done == True:
                print("existed " + path)
                continue
            # Skip scenes with missing data
            spp_file = os.path.join(cfg.data.spp_path, f"{scene_id}.pth")
            ply_file = os.path.join(cfg.data.original_ply, f"{scene_id}.ply")
            if not os.path.exists(spp_file) or not os.path.exists(ply_file):
                print(f"SKIP {scene_id}: missing data files")
                continue
            # # Write append each line
            with open(tracker_file, "a") as file:
                file.write(path + "\n")
            #####################################
            print("Process", scene_id)
            import time
            scene_start = time.time()
            proposals3d, mask2d_bank, id_bank, obj_bank = model.generate3dproposal(
                scene_id,
                cfg=cfg,
            )
            scene_end = time.time()
            print(f"\n[MASK_GENERATOR] Scene {scene_id} total time: {scene_end - scene_start:.2f}s\n")

            # Save 3D mask
            cluster_dict = {"ins": rle_encode_gpu_batch(proposals3d), 'mask2d_bank': mask2d_bank, 'id_bank': id_bank, 'obj_bank': obj_bank}
            torch.save(cluster_dict, os.path.join(save_dir_cluster, f"{scene_id}.pth"))            
            # Free memory
            torch.cuda.empty_cache()
