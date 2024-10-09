import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import kornia
import torch
import numpy as np

from .mano import mano


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

def vis_mano(pose_gt, pose_pred, shape_gt, shape_pred, hand_type: str = 'right'):
    assert hand_type in ['right', 'left']
    
    # 确保 pose_gt 和 pose_pred 的形状匹配
    batch_size = min(pose_gt.shape[0], 6)
    
    # 创建多行多列子图
    fig, axs = plt.subplots(batch_size, 2, subplot_kw={'projection': '3d'},
                            figsize=(10, 5 * batch_size))
    
    # 如果 batch_size 为 1，则 axs 可能会是 1 维的，需要处理这种情况
    if batch_size == 1:
        axs = [axs]

    for i in range(batch_size):
        # 生成 Mano 模型的 GT 和预测顶点
        output_gt = mano.layer[hand_type](betas=shape_gt[i:i+1],
                                          hand_pose=pose_gt[i:i+1, 3:],
                                          global_orient=pose_gt[i:i+1, :3],
                                          transl=torch.zeros(1, 3))
        output_pred = mano.layer[hand_type](betas=shape_pred[i:i+1],
                                            hand_pose=pose_pred[i:i+1, 3:],
                                            global_orient=pose_pred[i:i+1, :3],
                                            transl=torch.zeros(1, 3))
        
        faces = mano.face[hand_type]
        verts_gt = output_gt.vertices[0].numpy()
        verts_pred = output_pred.vertices[0].numpy()

        # 绘制 Ground Truth 手部模型
        mesh_gt = Poly3DCollection(verts_gt[faces], color='r', alpha=0.1, edgecolor='k')
        axs[i][0].add_collection3d(mesh_gt)
        axs[i][0].set_title(f'Batch {i+1} - Ground Truth')
        
        # 设置显示范围
        scale_gt = verts_gt.flatten()
        axs[i][0].auto_scale_xyz(scale_gt, scale_gt, scale_gt)

        # 绘制预测的手部模型
        mesh_pred = Poly3DCollection(verts_pred[faces], color='b', alpha=0.1, edgecolor='k')
        axs[i][1].add_collection3d(mesh_pred)
        axs[i][1].set_title(f'Batch {i+1} - Prediction')
        
        # 设置显示范围
        scale_pred = verts_pred.flatten()
        axs[i][1].auto_scale_xyz(scale_pred, scale_pred, scale_pred)

    # 渲染图像并将其转换为 NumPy 数组
    canvas = FigureCanvas(fig)
    canvas.draw()

    # 使用 buffer_rgba 代替 tostring_rgb
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA 通道

    plt.close(fig)  # 关闭图像，释放内存
    return img