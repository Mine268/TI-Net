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

def crop_img(img: torch.Tensor, bbox_center, bbox_size, squarify=True, avoid_zero=False):
    '''
    center, size 均遵从 interwild 工作中的定义，是一个二元组。第一、二个元素分别
    是水平方向和垂直方向的位置和长度。
    '''
    assert isinstance(img, torch.Tensor), "Only torch.Tensor image is supported"
    
    w_center, h_center = bbox_center
    width, height = bbox_size

    if squarify:
        length = max(width, height)
        width = length
        height = length
    if avoid_zero:
        width = max(width, 2)  # ! use 2 instead of 1
        height = max(height, 2)
    
    w_min = (w_center - width / 2)
    h_min = (h_center - height / 2)
    w_max = (w_center + width / 2)
    h_max = (h_center + height / 2)
    boxes = torch.tensor([[
        [w_min, h_min], [w_max, h_min], [w_max, h_max], [w_min, h_max]
    ]])
    output_size = (int(height), int(width))

    cropped_img = kornia.geometry.transform.crop_and_resize(img[None,...], boxes, output_size)
    return cropped_img[0]

def compute_pa_mpjpe_batch(pred_batch: torch.Tensor, gt_batch: torch.Tensor):
    """
    计算批量版本的 PA-MPJPE
    :param pred_batch: 预测的点集，形状为 [B, J, 3]
    :param gt_batch: 真实的点集，形状为 [B, J, 3]
    :return: PA-MPJPE，形状为 [B] 的误差向量
    """
    # Step 1: 中心化
    pred_mean = pred_batch.mean(dim=1, keepdim=True)  # [B, 1, 3]
    gt_mean = gt_batch.mean(dim=1, keepdim=True)      # [B, 1, 3]
    
    pred_centered = pred_batch - pred_mean            # [B, J, 3]
    gt_centered = gt_batch - gt_mean                  # [B, J, 3]

    # Step 2: SVD 分解求旋转矩阵
    # 将 B 维度和 J 维度展开为矩阵乘法，分别计算每个批次的旋转矩阵
    H = torch.einsum('bij,bik->bjk', pred_centered, gt_centered)  # [B, 3, 3]
    U, S, Vt = torch.svd(H)
    
    R = torch.matmul(Vt, U.transpose(1, 2))  # [B, 3, 3]
    
    # 防止 R 是反射矩阵而不是旋转矩阵
    det_R = torch.det(R)  # [B]
    Vt[det_R < 0, :, 2] *= -1  # 修改反射矩阵
    R = torch.matmul(Vt, U.transpose(1, 2))

    # Step 3: 计算缩放因子 s
    scale = torch.einsum('bij,bij->b', torch.matmul(pred_centered, R), gt_centered).sum(dim=-1) / (pred_centered ** 2).sum(dim=[1, 2])

    # Step 4: 应用旋转和缩放
    pred_aligned = scale[:, None, None] * torch.matmul(pred_centered, R)  # [B, J, 3]

    # Step 5: 平移对齐
    pred_aligned += gt_mean

    # Step 6: 计算 PA-MPJPE (欧氏距离)
    error = torch.norm(pred_aligned - gt_batch, dim=-1)  # [B, J]
    pa_mpjpe = error.mean(dim=-1)  # [B]

    return pa_mpjpe
