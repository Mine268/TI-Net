from typing import *
import os
import json
import einops as eps

import kornia
import cv2
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


class FreiHand(Dataset):
    def __init__(self, mode, root="./data/freihand"):
        assert mode in ["train", "val"]
        self.mode = mode
        if mode == "train":
            self.data_root = os.path.join(root, "training")
            self.img_root = os.path.join(self.data_root, "rgb")
            self.intr_file = os.path.join(root, "training_K.json")
            self.kps_file = os.path.join(root, "training_xyz.json")
            self.mano_file = os.path.join(root, "training_mano.json")
        else:
            self.data_root = os.path.joint(root, "evaluation")
            self.img_root = os.path.join(self.data_root, "rgb")
            self.intr_file = os.path.join(root, "evaluation_K.json")
            self.kps_file = os.path.join(root, "evaluation_xyz.json")
            self.mano_file = os.path.join(root, "evaluation_mano.json")

        self.image_preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        with open(self.intr_file, "r") as f:
            self.intrs = json.load(f)
            self.intrs = torch.Tensor(self.intrs)
        with open(self.kps_file, "r") as f:
            self.kps3d = json.load(f)
            self.kps3d = torch.Tensor(self.kps3d)
        with open(self.mano_file, "r") as f:
            self.mano_data = json.load(f)
            self.mano_data = torch.Tensor(self.mano_data)
            self.pose, self.shape, self.uv_root, self.scale = \
                eps.unpack(self.mano_data, [(48,), (10,), (2,), (1,)], "b d *")
        
        self.len = self.kps3d.shape[0]

    def __len__(self) -> int:
        return self.len 
    
    def __getitem__(self, ix: int) -> Tuple:
        img = cv2.imread(os.path.join(self.img_root, f"{ix:08d}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # bgr -> rgb
        img = self.image_preprocessing(img)

        kps3d = self.kps3d[ix]
        intr = self.intrs[ix]
        pose = self.pose[ix, 0]
        shape = self.shape[ix, 0]
        uv_root = self.uv_root[ix, 0]
        scale = self.scale[ix, 0]

        # rotation transformation
        if self.mode == "train":
            rot_deg = torch.rand(size=(1,)) * torch.pi * 2
            rot_mat = torch.Tensor([
                [torch.cos(-rot_deg), -torch.sin(-rot_deg), 0],
                [torch.sin(-rot_deg), torch.cos(-rot_deg), 0],
                [0, 0, 1]
            ])
            root_pose = pose[:3]
            root_pose, _ = cv2.Rodrigues(root_pose.numpy())
            root_pose, _ = cv2.Rodrigues(rot_mat.numpy() @ root_pose)
            pose[:3] = torch.from_numpy(root_pose.reshape(3))

            kps3d = (rot_mat @ kps3d.T).T

            rot_ang = rot_deg * 180 / torch.pi
            img = kornia.geometry.transform.rotate(img[None,...], rot_ang)[0]

        return {"image": img,
                "K": intr,
                "pose": pose,
                "shape": shape,
                "uv_root": uv_root,
                "scale": scale,
                "joints_3d": kps3d}


if __name__ == "__main__":
    fh_dataset = FreiHand("train", root="/mnt/data_0/renkaiwen/sl_vit/data/freihand")
    print(len(fh_dataset))
    item = fh_dataset[4993]
    pass
