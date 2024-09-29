import copy
from datetime import datetime
from typing import *
import os
import argparse

import einops as eps
import torch
import torch.utils
import torch.nn as nn
import numpy as np
import einops as eps
from tqdm import tqdm

from dataset import FreiHand 
# from dataset.InterHand26M.utils.mano import mano
from utils import mano
import resnet
import vit
from utils import compute_pa_mpjpe_batch


def parse_arg():
    parser = argparse.ArgumentParser("Testing")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--model", default="resnet/pose_resnet50", type=str)
    parser.add_argument("--ckpt", default=None, required=True, type=str)
    parser.add_argument("--output_dir", default="./logs/debug")
    parser.add_argument("--device", default=0, type=int, help="Only support single GPU inference")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--norm_pix_loss", default=True, type=bool)
    
    args = parser.parse_args()
    return args


joint_raw_regressor = None
def eval_batch(mano_layer,
               pose_pred: torch.Tensor, pose_gt: torch.Tensor, shape_gt: torch.Tensor) \
                   -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = pose_pred.shape[0]
    pose_pred = pose_pred.reshape(batch_size, -1)
    pose = torch.cat([pose_pred, pose_gt], dim=0)
    shape = torch.cat([shape_gt, shape_gt], dim=0) 
    output = mano_layer['right'](betas=shape,
                                 hand_pose=pose[:,3:],
                                 global_orient=pose[:,:3],
                                 transl=torch.zeros(batch_size * 2, 3, device=pose.device))

    global joint_raw_regressor
    if joint_raw_regressor is None:
        joint_raw_regressor = torch.from_numpy(mano.sh_joint_regressor).to(pose_gt.device)
    joint_regressor = joint_raw_regressor[None,...].repeat(batch_size, 1, 1)
        
    mesh_pred, mesh_gt = eps.unpack(output.vertices, [(batch_size,), (batch_size,)], "* v d")
    joint_pred = torch.bmm(joint_regressor, mesh_pred)
    joint_gt = torch.bmm(joint_regressor, mesh_gt)

    mesh_error = ((mesh_gt - mesh_pred) ** 2).sum(-1, keepdim=True).sqrt()
    joint_error = ((joint_gt - joint_pred) ** 2).sum(-1, keepdim=True).sqrt()
    joint_pa_error = compute_pa_mpjpe_batch(joint_pred, joint_gt)

    return mesh_error, joint_error, joint_pa_error


def test(args):
    mano_layer = copy.deepcopy(mano.layer)  # .to(f"cuda:{args.device}")
    mano_layer['left'].to(f"cuda:{args.device}")
    mano_layer['right'].to(f"cuda:{args.device}")

    dataset_train = FreiHand(mode="test") 
    dataloader = torch.utils.data.DataLoader(dataset_train,
                                             batch_size=args.batch_size,
                                             pin_memory=True,
                                             drop_last=False)

    model_str = args.model
    model_class, model_arch = model_str.split('/')
    if model_class == 'vit':
        model = vit.__dict__[model_arch](norm_pix_loss=args.norm_pix_loss)
        # TODO: load checkpoint from the file
    elif model_class == 'resnet':
        model: nn.Module = resnet.__dict__[model_arch]()
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt['model'])
    else:
        assert False, "model not supported: %s" % model_str
    model.to(f"cuda:{args.device}")
    model.eval()

    mpjpe_list = []
    pampjpe_list = []

    for _, data_item in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100):
        images = data_item["image"].to(args.device)
        pose_gt = data_item["pose"].to(args.device)
        shape_gt = data_item["shape"].to(args.device)
        
        with torch.no_grad():
            pred = model(images)
            mesh_error, joint_error, joint_pa_error = eval_batch(mano_layer, pred, pose_gt, shape_gt)

        mpjpe_list.extend(joint_error.tolist())
        pampjpe_list.extend(joint_pa_error.tolist())

    mpjpe = np.mean(mpjpe_list)
    pampjpe = np.mean(pampjpe_list)

    print(f"MPJPE: {mpjpe} m")
    print(f"PA-MPJPE: {pampjpe} m")

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(args.output_dir, f"eval_Freihand_{time_str}.txt"), "w") as f:
        f.write(f"MPJPE: {mpjpe} m\nPAMPJPE: {pampjpe}")


if __name__ == "__main__":
    args = parse_arg()
    test(args)
