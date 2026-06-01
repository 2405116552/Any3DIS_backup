import os

import numpy as np
import torch
from loader3d.scannet200 import INSTANCE_CAT_SCANNET_200
from loader3d.scannetpp import SEMANTIC_CAT_SCANNET_PP, INSTANCE_BENCHMARK84_SCANNET_PP, SEMANTIC_INSTANCE_CAT_SCANNET_PP # ScannetPP
from scannetv2_inst_eval import ScanNetEval
from tqdm import tqdm
import argparse
import yaml
from munch import Munch

def rle_decode(rle):
    length = rle["length"]
    try:
        s = rle["counts"].split()
    except:
        s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def rle_decode(rle):
    length = rle["length"]
    s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    parser.add_argument("--type",type=str,required = True,help="[2D, 3D, 2D_3D]") # raw 3DIS

    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    eval_type= args.type
    
    if cfg.data.dataset_name  == 'scannet200':
        scan_eval = ScanNetEval(class_labels=INSTANCE_CAT_SCANNET_200, use_label = False, dataset_name = 'scannet200')
        pcl_path = cfg.data.gt_pth # groundtruth
        if eval_type == '2D':
            data_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
        if eval_type == '3D':
            data_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path)
        if eval_type == '2D_3D':
            pass
    if cfg.data.dataset_name  == 'scannetpp': # 
        # eval on 1554instance classes, inputting sem+ins set
        scan_eval = ScanNetEval(class_labels=SEMANTIC_INSTANCE_CAT_SCANNET_PP, use_label = False, dataset_name = 'scannetpp')
        pcl_path = cfg.data.gt_pth # groundtruth
        if eval_type == '2D':
            data_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
        if eval_type == '3D':
            data_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path)
        if eval_type == '2D_3D':
            pass
    if cfg.data.dataset_name  == 'scannetpp_benchmark': # 
        # eval on 84 top instance classes
        scan_eval = ScanNetEval(class_labels=INSTANCE_BENCHMARK84_SCANNET_PP, use_label = False, dataset_name = 'scannetpp_benchmark')
        pcl_path = cfg.data.gt_pth # groundtruth
        if eval_type == '2D':
            data_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
        if eval_type == '3D':
            data_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path)
        if eval_type == '2D_3D':
            pass


    gtsem = []
    gtinst = []
    res = [] #ScannetV2
    

    # data_path already set from config above
    # data_path = "../freevocab_exp_scannetpp/version_sai3d/mask2d_lifted"

    # data_path3D = "./data/Scannet200/Scannet200_3D/class_ag_res_200_isbnetfull"

    scenes = os.listdir(data_path)

    # scenes = ['0d2ee665be.pth']
    for scene in tqdm(scenes):
        scene_path = os.path.join(data_path, scene)
        pred_mask = torch.load(scene_path)
        # pred_mask3D = torch.load(scene_path3D)
        # pred_mask1 = torch.load(scene_path1)


        gt_path = os.path.join(pcl_path, scene)
        loader = torch.load(gt_path)
        sem_gt, inst_gt = loader[2], loader[3]
        gtsem.append(np.array(sem_gt).astype(np.int32))
        gtinst.append(np.array(inst_gt).astype(np.int32))
        
        masks = pred_mask['ins']        
        
        n_mask = len(masks)
        tmp = []
        
        # masks3D = pred_mask3D['ins']
        # for mask in masks3D:
        #     conf = 1.0
        #     scene_id = scene.replace('.pth', '')
        #     tmp.append({"scan_id": scene_id, "label_id": 0, "conf": conf, "pred_mask": mask}) # class-agnostic evaluation

        for ind in range(n_mask):
            if isinstance(masks[ind], dict):
                mask = rle_decode(masks[ind])
            else:
                try:
                    mask = (masks[ind] == 1).numpy().astype(np.uint8)
                except:
                    mask = (masks[ind] == 1).astype(np.uint8)

            # conf = score[ind] #
            conf = 1.0

            scene_id = scene.replace('.pth', '')
            tmp.append({"scan_id": scene_id, "label_id": 0, "conf": conf, "pred_mask": mask}) # class-agnostic evaluation
        res.append(tmp)

    scan_eval.evaluate(res, gtsem, gtinst)
