# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from .pos_embed import get_2d_sincos_pos_embed
import utils


ROT_EMBED_DIM = 32
LOW_RANK = 128

class TI_ViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 train_secondary_trans=True):
        super().__init__()
        self.t_num = 14 if train_secondary_trans else 4
        self.train_secondary_trans = train_secondary_trans

        # --------------------------------------------------------------------------
        # Isomorphic Transformations
        self.rot_embed = nn.Sequential(nn.Linear(2, ROT_EMBED_DIM, bias=True), nn.ReLU(),
                                       nn.Linear(ROT_EMBED_DIM, ROT_EMBED_DIM, bias=True), nn.ReLU(),
                                       nn.Linear(ROT_EMBED_DIM, ROT_EMBED_DIM, bias=False))
        self.horizontal_flip = nn.Sequential(nn.Linear(embed_dim, LOW_RANK, bias=False),
                                             nn.Linear(LOW_RANK, embed_dim, bias=False))
        self.rotation = nn.Sequential(nn.Linear(embed_dim + ROT_EMBED_DIM, LOW_RANK, bias=False),
                                      nn.Linear(LOW_RANK, embed_dim, bias=False))
        self.horizontal_rot = nn.Sequential(nn.Linear(embed_dim + ROT_EMBED_DIM, LOW_RANK, bias=False),
                                            nn.Linear(LOW_RANK, embed_dim, bias=False))
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # cls_token 直接使用 patch_token 的平均
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)    

    def latent_operate(self, operate: str, latent: torch.Tensor, rot: torch.Tensor = None):
        '''
        operate:
         - h: horizontal flip, rot=None
         - r: rotation, rot required
         - hr: hflip & rot, rot required
        
        latent: [B D]
        
        rot: [1] / [B]
        '''
        assert operate in ['h', 'r', 'hr'], f"operate '{operate}' is not supported."
        assert not (operate in ['r', 'hr'] and rot is None), f"rot required."

        if operate == 'h':
            return latent + self.horizontal_flip(latent)
        
        # encode rotation
        rot = rot if rot.shape[0] == latent.shape[0] else rot.repeat(latent.shape[0])
        rot.to(latent.device)
        rot_vec = torch.cat([torch.cos(rot)[...,None], torch.sin(rot)[...,None]], dim=-1)
        rot_embedding = self.rot_embed(rot_vec)
        
        if operate == 'r':
            return latent + self.rotation(torch.cat([latent, rot_embedding], dim=-1))
        
        if operate == 'hr':
            return latent + self.horizontal_rot(torch.cat([latent, rot_embedding], dim=-1))

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 0:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, cls_token, ids_restore):
        # embed tokens
        cls_token = self.decoder_embed(cls_token)
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([cls_token, x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, latent, a1, a2, b1, b2):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # ----------------------------------------------------------------------
        # 1. Reconstruction loss
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        recon_loss = (pred - target) ** 2
        recon_loss = recon_loss.mean(dim=-1)  # [N, L], mean loss per patch

        recon_loss = (recon_loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # ----------------------------------------------------------------------
        
        # ----------------------------------------------------------------------
        # 2. Isomorphism loss
        # (0) Identity operation
        loss_iden = F.mse_loss(self.latent_operate('r', latent[0], torch.zeros_like(a1)), latent[0])
        # (1) Uniary operation
        loss_uni_h = F.mse_loss(self.latent_operate('h', latent[0]), latent[1])
        if self.train_secondary_trans:
            loss_uni_r = F.mse_loss(self.latent_operate('r', latent[0], a1), latent[2]) + F.mse_loss(self.latent_operate('r', latent[0], b1), latent[4])
            loss_uni_hr = F.mse_loss(self.latent_operate('hr', latent[0], a2), latent[3]) + F.mse_loss(self.latent_operate('hr', latent[0], b2), latent[5])
        else:
            loss_uni_r = F.mse_loss(self.latent_operate('r', latent[0], a1), latent[2])
            loss_uni_hr = F.mse_loss(self.latent_operate('hr', latent[0], a2), latent[3])
        loss_uni = loss_uni_h + loss_uni_r + loss_uni_hr
        # (2) Binary operation
        if self.train_secondary_trans:
            loss_bin_h_h = F.mse_loss(self.latent_operate('h', self.latent_operate('h', latent[0])), latent[0])
            loss_bin_h_r = F.mse_loss(self.latent_operate('r', self.latent_operate('h', latent[0]), b1), latent[6])
            loss_bin_h_hr = F.mse_loss(self.latent_operate('hr', self.latent_operate('h', latent[0]), b2), latent[8])
            loss_bin_r_h = F.mse_loss(self.latent_operate('h', self.latent_operate('r', latent[0], a1)), latent[7])
            loss_bin_r_r = F.mse_loss(self.latent_operate('r', self.latent_operate('r', latent[0], a1), b1), latent[9])
            loss_bin_r_hr = F.mse_loss(self.latent_operate('hr', self.latent_operate('r', latent[0], a1), b2), latent[11])
            loss_bin_hr_h = F.mse_loss(self.latent_operate('h', self.latent_operate('hr', latent[0], a2)), latent[10])
            loss_bin_hr_r = F.mse_loss(self.latent_operate('r', self.latent_operate('hr', latent[0], a2), b1), latent[12])
            loss_bin_hr_hr = F.mse_loss(self.latent_operate('hr', self.latent_operate('hr', latent[0], a2), b2), latent[13])
            loss_bin = loss_bin_h_h + loss_bin_h_r + loss_bin_h_hr + loss_bin_r_h + loss_bin_r_r + \
                       loss_bin_r_hr + loss_bin_hr_h + loss_bin_hr_r + loss_bin_hr_hr
        # (3) Total loss
        if self.train_secondary_trans:
            latent_loss = loss_iden + loss_uni + loss_bin
        else:
            latent_loss = loss_iden + loss_uni
        # ----------------------------------------------------------------------

        loss = recon_loss + 1e-3 * latent_loss
        return {
            "backward": loss,
            "reconstruction": recon_loss,
            "latent": latent_loss
        }

    def forward(self, imgs, mask_ratio=0.75):
        '''
        imgs [B,3,H,W]
        '''
        a1 = torch.rand(size=(imgs.shape[0],), device=imgs.device) * 2 * torch.pi
        a2 = torch.rand(size=(imgs.shape[0],), device=imgs.device) * 2 * torch.pi
        b1 = torch.rand(size=(imgs.shape[0],), device=imgs.device) * 2 * torch.pi
        b2 = torch.rand(size=(imgs.shape[0],), device=imgs.device) * 2 * torch.pi

        # ordinary transformation
        imgs_h = utils.horizontal_flip_img(imgs).detach()
        imgs_r_a1 = utils.rotate_img(imgs, a1).detach()
        imgs_hr_a2 = utils.hflip_rotate_img(imgs, a2).detach()
        imgs_r_b1 = utils.rotate_img(imgs, b1).detach()
        imgs_hr_b2 = utils.hflip_rotate_img(imgs, b2).detach()

        # secondary transformation
        imgs_hr_b1 = utils.hflip_rotate_img(imgs, b1).detach()
        imgs_hr_na1 = utils.hflip_rotate_img(imgs, -a1).detach()
        imgs_r_b2 = utils.rotate_img(imgs, b2).detach()
        imgs_r_a1_b1 = utils.rotate_img(imgs, a1 + b1).detach()
        imgs_r_na2 = utils.rotate_img(imgs, -a2).detach()
        imgs_hr_na1_b2 = utils.hflip_rotate_img(imgs, -a1 + b2).detach()
        imgs_hr_a2_b1 = utils.hflip_rotate_img(imgs, a2 + b1).detach()
        imgs_r_na2_b2 = utils.rotate_img(imgs, -a2 + b2).detach()

        # batch-up images
        imgs_batch = einops.rearrange(
            torch.cat([imgs[:,None],  # 0
                       imgs_h[:,None],
                       imgs_r_a1[:,None],  # 2
                       imgs_hr_a2[:,None],
                       imgs_r_b1[:,None],  # 4
                       imgs_hr_b2[:,None],
                       imgs_hr_b1[:,None],  # 6
                       imgs_hr_na1[:,None],
                       imgs_r_b2[:,None],  # 8
                       imgs_r_a1_b1[:,None],
                       imgs_r_na2[:,None],  # 10
                       imgs_hr_na1_b2[:,None],
                       imgs_hr_a2_b1[:,None],  # 12
                       imgs_r_na2_b2[:,None]],
                       dim=1)[:,:self.t_num],
            'b t ... -> (b t) ...', t=self.t_num
        )

        latents, masks, ids_restores = self.forward_encoder(imgs_batch, mask_ratio)
        cls_tokens = torch.mean(latents, dim=1, keepdim=False)
        preds = self.forward_decoder(latents, cls_tokens[:,None], ids_restores)
        losses = self.forward_loss(imgs_batch, preds, masks,
                                   einops.rearrange(cls_tokens, '(b t) ... -> t b ...', t=self.t_num),
                                   a1, a2, b1, b2)

        pred_0 = einops.rearrange(preds, '(b t) ... -> b t ...', t=self.t_num)
        pred_0 = pred_0[:, 0].detach()
        pred_0 = self.unpatchify(pred_0)

        return losses, pred_0
