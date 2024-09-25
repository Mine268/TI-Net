import copy
from datetime import datetime
import os
import argparse

import einops as eps
import torch
import torch.utils
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import einops as eps
from tqdm import tqdm

from dataset.InterHand26M import InterHand26M
from dataset.InterHand26M.utils.mano import mano
import resnet
import vit


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

joint_regressor = None
def eval_batch(mano_layer, pred: torch.Tensor, targets: dict, meta_info: dict) -> list:
    mano_pose_pred = eps.rearrange(pred, '(h b) j d -> h b j d', h=2)
    rmano_pose_pred = mano_pose_pred[0]
    lmano_pose_pred = mano_pose_pred[1]
    lmano_pose_pred = torch.cat([lmano_pose_pred[:,:,0:1], -lmano_pose_pred[:,:,1:3]], dim=2)

    mano_pose_gt = targets['mano_pose']
    mano_shape_gt = targets['mano_shape']
    rmano_pose_gt = mano_pose_gt[:,:48]
    lmano_pose_gt = mano_pose_gt[:,48:]
    rmano_shape_gt = mano_shape_gt[:,:10]
    lmano_shape_gt = mano_shape_gt[:,10:]

    batch_size = mano_pose_pred.shape[1]
    rmano_pose = torch.cat([rmano_pose_pred.view(batch_size, -1), rmano_pose_gt], dim=0)
    rmano_shape = eps.repeat(rmano_shape_gt, 'b d -> (r b) d', r=2)
    lmano_pose = torch.cat([lmano_pose_pred.view(batch_size, -1), lmano_pose_gt], dim=0)
    lmano_shape = eps.repeat(lmano_shape_gt, 'b d -> (r b) d', r=2)

    r_output = mano_layer['right'](betas=rmano_shape,
                                   hand_pose=rmano_pose[:,3:],
                                   global_orient=rmano_pose[:,:3],
                                   transl=torch.zeros(batch_size*2, 3, device=rmano_shape.device))
    l_output = mano_layer['left'](betas=lmano_shape,
                                  hand_pose=lmano_pose[:,3:],
                                  global_orient=lmano_pose[:,:3],
                                  transl=torch.zeros(batch_size*2, 3, device=lmano_shape.device))

    rmesh_pred = r_output.vertices[:batch_size]
    rmesh_gt = r_output.vertices[batch_size:]
    lmesh_pred = l_output.vertices[:batch_size]
    lmesh_gt = l_output.vertices[batch_size:]

    global joint_regressor
    if joint_regressor is None:
        joint_regressor = torch.from_numpy(mano.sh_joint_regressor)[None,...].repeat(batch_size,1,1).to(rmano_shape.device)
    rjoint_pred = torch.bmm(joint_regressor, rmesh_pred)
    rjoint_gt = torch.bmm(joint_regressor, rmesh_gt)
    ljoint_pred = torch.bmm(joint_regressor, lmesh_pred)
    ljoint_gt = torch.bmm(joint_regressor, lmesh_gt)

    mesh_pred = torch.cat([rmesh_pred, lmesh_pred], dim=1)
    mesh_gt = torch.cat([rmesh_gt, lmesh_gt], dim=1)
    joint_pred = torch.cat([rjoint_pred, ljoint_pred], dim=1)
    joint_gt = torch.cat([rjoint_gt, ljoint_gt], dim=1)

    mesh_error = ((mesh_pred - mesh_gt) ** 2).sum(-1, keepdim=True).sqrt()
    joint_error = ((joint_pred - joint_gt) ** 2).sum(-1, keepdim=True).sqrt()
    mesh_valid = meta_info['mano_mesh_valid']
    joint_valid = meta_info['joint_valid']

    return mesh_error, mesh_valid, joint_error, joint_valid


def test(args):
    mano_layer = copy.deepcopy(mano.layer)  # .to(f"cuda:{args.device}")
    mano_layer['left'].to(f"cuda:{args.device}")
    mano_layer['right'].to(f"cuda:{args.device}")

    transforms_test = transforms.Compose([
        transforms.Resize((224, 224))
    ])
    dataset_train = InterHand26M(transforms_test, "test")
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, pin_memory=True, drop_last=True)

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
    mpvpe_sum, vertix_num = 0.0, 0

    for ix, (inputs_, targets_, meta_info_) in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100):
        rhand_img = inputs_['rhand_img']
        lhand_img = torch.flip(inputs_['lhand_img'], dims=[3])
        samples = torch.concatenate([rhand_img, lhand_img], dim=0)
        samples = samples.to(f"cuda:{args.device}")

        # targets to cuda
        for key, value in targets_.items():
            targets_[key] = value.to(args.device, non_blocking=True)
        # meta info to cuda
        for key, value in meta_info_.items():
            meta_info_[key] = value.to(args.device, non_blocking=True)

        with torch.no_grad():
            pred = model(samples)
            mesh_error, mesh_valid, joint_error, joint_valid = eval_batch(mano_layer, pred, targets_, meta_info_)

        mpjpe_list.extend(joint_error[joint_valid > 0.5].tolist())
        cur_mpvpe = mesh_error[mesh_valid > 0.5]
        mpvpe_sum += cur_mpvpe.sum().item()
        vertix_num += cur_mpvpe.shape[0]

    mpjpe = np.mean(mpjpe_list)
    mpvpe = mpvpe_sum / vertix_num

    print(f"MPJPE: {mpjpe} m")
    print(f"MPVPE: {mpvpe} m")

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(args.output_dir, f"eval_InterHand26M_{time_str}.txt"), "w") as f:
        f.write(f"MPJPE: {mpjpe} m\nMPVPE: {mpvpe} m")


if __name__ == "__main__":
    test(parse_arg())
