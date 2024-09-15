import numpy as np
import cv2
import kornia
import torch


def horizontal_flip_img(imgs: torch.Tensor) -> torch.Tensor:
    '''
    imgs: [B,C,H,W]
    '''
    return torch.flip(imgs, dims=[3])

def rotate_img(imgs: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
    '''
    imgs: [B,C,H,W]
    degree: [B]
    '''
    center = torch.tensor([imgs.shape[2] / 2, imgs.shape[3] / 2], device=imgs.device).repeat(imgs.size(0), 1)
    M = kornia.geometry.transform.get_rotation_matrix2d(center, degree, torch.ones_like(center))
    rotated_imgs = kornia.geometry.transform.warp_affine(imgs, M, (imgs.shape[2], imgs.shape[3]))
    return rotated_imgs

def hflip_rotate_img(imgs: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
    '''
    imgs: [B,C,H,W]
    degree: [B]
    '''
    flipped_imgs = horizontal_flip_img(imgs)
    rotated_flipped_imgs = rotate_img(flipped_imgs, degree)
    return rotated_flipped_imgs
