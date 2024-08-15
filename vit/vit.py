# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import Block, PatchEmbed
import einops

from pos_embed import *


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


# ref: MaskedAutoencoderViT
class VitAutoEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        
        self.patch_size = patch_size

        # --------------------------------------------------------------------------
        # ViTAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # ViTAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

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
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

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
    
    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    
    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

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

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = loss.sum(-1).mean()
        return {"backward": loss}

    def forward(self, imgs):
        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred)
        return loss, self.unpatchify(pred).detach()


# ref: MaskedAutoencoderViT
class SL_VitAutoEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        
        # --------------------------------------------------------------------------
        # Structural latent, auxliary
        self.operator_embeddings = nn.Parameter(torch.randn(3, embed_dim), requires_grad=True)
        self.latent_operator = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(3)
        ])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # ViTAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # ViTAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

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
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

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
        
    def latent_operate(self, operator_ix, latent):
        ''' latent [N,L,D]
        '''
        np = self.img_size // self.patch_size
        operator_emb = self.operator_embeddings[operator_ix]

        latent = einops.rearrange(latent, 'b (p q) d -> b p q d', p=np, q=np)
        if operator_ix == 0:
            latent = torch.flip(latent, dims=[2])
        if operator_ix == 1:
            latent = torch.flip(latent, dims=[3])
        if operator_ix == 2:
            latent = torch.flip(latent, dims=[2,3])
        latent = einops.rearrange(latent, 'b p q d -> b (p q) d', p=np, q=np)

        input_ = latent + operator_emb[None, None, :]
        for blk in self.latent_operator:
            input_ = blk(input_)
        output_ = self.norm(input_)
        return output_
        
    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    
    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

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

    def forward_loss(self, imgs, pred, latent):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        """
        # TODO: token 顺序调整
        # --------------------------------------------------------------------------
        # Reconstruction loss
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        recon_loss = (pred - target) ** 2
        recon_loss = recon_loss.mean(dim=-1)  # [N, L], mean loss per patch
        recon_loss = recon_loss.sum(-1).mean()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Structural latent loss
        latent = einops.rearrange(latent, "(b t) l d -> t b l d", t=4)  # t: 4 types of operators
        latent = latent[:,:,1:]
        # 1. Uniary operation
        loss_uni_0 = F.mse_loss(latent[1], self.latent_operate(0, latent[0])) + F.mse_loss(latent[0], self.latent_operate(0, latent[1])) + \
                        F.mse_loss(latent[3], self.latent_operate(0, latent[2])) + F.mse_loss(latent[2], self.latent_operate(0, latent[3]))
        loss_uni_1 = F.mse_loss(latent[2], self.latent_operate(1, latent[0])) + F.mse_loss(latent[0], self.latent_operate(1, latent[2])) + \
                        F.mse_loss(latent[1], self.latent_operate(1, latent[3])) + F.mse_loss(latent[3], self.latent_operate(1, latent[1]))
        loss_uni_2 = F.mse_loss(latent[0], self.latent_operate(2, latent[3])) + F.mse_loss(latent[3], self.latent_operate(2, latent[0])) + \
                        F.mse_loss(latent[1], self.latent_operate(2, latent[2])) + F.mse_loss(latent[2], self.latent_operate(2, latent[1]))
        loss_uni = loss_uni_0 + loss_uni_1 + loss_uni_2
        # 2. Binary operation
        loss_bin_hh = F.mse_loss(self.latent_operate(0, self.latent_operate(0, latent[0])), latent[0])
        loss_bin_hv = F.mse_loss(self.latent_operate(1, self.latent_operate(0, latent[0])), latent[3])
        loss_bin_hc = F.mse_loss(self.latent_operate(2, self.latent_operate(0, latent[0])), latent[2])
        loss_bin_vh = F.mse_loss(self.latent_operate(0, self.latent_operate(1, latent[0])), latent[3])
        loss_bin_vv = F.mse_loss(self.latent_operate(1, self.latent_operate(2, latent[0])), latent[0])
        loss_bin_vc = F.mse_loss(self.latent_operate(2, self.latent_operate(1, latent[0])), latent[1])
        loss_bin_ch = F.mse_loss(self.latent_operate(0, self.latent_operate(2, latent[0])), latent[2])
        loss_bin_cv = F.mse_loss(self.latent_operate(1, self.latent_operate(2, latent[0])), latent[1])
        loss_bin_cc = F.mse_loss(self.latent_operate(2, self.latent_operate(2, latent[0])), latent[0])
        loss_bin = loss_bin_hh + loss_bin_hv + loss_bin_hc + loss_bin_vh + loss_bin_vv + loss_bin_vc + loss_bin_ch + loss_bin_cv + loss_bin_cc
        # 3. Total loss
        latent_loss = loss_uni + loss_bin
        # --------------------------------------------------------------------------

        loss = recon_loss + latent_loss * 0.001
        return {"backward": loss, "reconstruction": recon_loss, "latent": latent_loss}

    def forward(self, imgs):
        ''' imgs: [N,3,H,W]
        '''
        imgs_vf = torch.flip(imgs, dims=[2])
        imgs_hf = torch.flip(imgs, dims=[3])
        imgs_cm = torch.flip(imgs, dims=[2,3])

        imgs_bundle = torch.cat([imgs[:,None], imgs_vf[:,None], imgs_hf[:,None], imgs_cm[:,None]], dim=1)
        imgs_batch = einops.rearrange(imgs_bundle, 'b t c h w -> (b t) c h w', t=4)

        latent = self.forward_encoder(imgs_batch)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs_batch, pred, latent)

        pred_0 : torch.Tensor = einops.rearrange(pred, '(b t) l d -> b t l d', t=4).detach()
        pred_0 = pred_0[:,0]

        return loss, self.unpatchify(pred_0)


# ref: MaskedAutoencoderViT
class SLL_VitAutoEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        
        # --------------------------------------------------------------------------
        # Structural latent, auxliary
        self.vf_operator = nn.Linear(embed_dim, embed_dim, bias=False)
        self.hf_operator = nn.Linear(embed_dim, embed_dim, bias=False)
        self.cm_operator = nn.Linear(embed_dim, embed_dim, bias=False)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # ViTAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # ViTAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

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
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

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
        
    def latent_operate(self, operator_ix, latent):
        ''' latent [N,L,D]
        '''
        np = self.img_size // self.patch_size
        latent = einops.rearrange(latent, 'b (p q) d -> b p q d', p=np, q=np)
        if operator_ix == 0:
            latent = torch.flip(latent, dims=[2])
            latent = self.vf_operator(latent)
        if operator_ix == 1:
            latent = torch.flip(latent, dims=[3])
            latent = self.hf_operator(latent)
        if operator_ix == 2:
            latent = torch.flip(latent, dims=[2,3])
            latent = self.cm_operator(latent)
        latent = einops.rearrange(latent, 'b p q d -> b (p q) d', p=np, q=np)
        return latent
        
    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    
    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

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

    def forward_loss(self, imgs, pred, latent):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        """
        # TODO: 只对原始图像重建误差 
        # --------------------------------------------------------------------------
        # Reconstruction loss
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        recon_loss = (pred - target) ** 2
        recon_loss = recon_loss.mean(dim=-1)  # [N, L], mean loss per patch
        recon_loss = recon_loss.sum(-1).mean()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Structural latent loss
        latent = einops.rearrange(latent, "(b t) l d -> t b l d", t=4)  # t: 4 types of operators
        latent = latent[:,:,1:]
        # 1. Uniary operation
        loss_uni_0 = F.mse_loss(latent[1], self.latent_operate(0, latent[0])) + F.mse_loss(latent[0], self.latent_operate(0, latent[1])) + \
                        F.mse_loss(latent[3], self.latent_operate(0, latent[2])) + F.mse_loss(latent[2], self.latent_operate(0, latent[3]))
        loss_uni_1 = F.mse_loss(latent[2], self.latent_operate(1, latent[0])) + F.mse_loss(latent[0], self.latent_operate(1, latent[2])) + \
                        F.mse_loss(latent[1], self.latent_operate(1, latent[3])) + F.mse_loss(latent[3], self.latent_operate(1, latent[1]))
        loss_uni_2 = F.mse_loss(latent[0], self.latent_operate(2, latent[3])) + F.mse_loss(latent[3], self.latent_operate(2, latent[0])) + \
                        F.mse_loss(latent[1], self.latent_operate(2, latent[2])) + F.mse_loss(latent[2], self.latent_operate(2, latent[1]))
        loss_uni = loss_uni_0 + loss_uni_1 + loss_uni_2
        # 2. Binary operation
        loss_bin_hh = F.mse_loss(self.latent_operate(0, self.latent_operate(0, latent[0])), latent[0])
        loss_bin_hv = F.mse_loss(self.latent_operate(1, self.latent_operate(0, latent[0])), latent[3])
        loss_bin_hc = F.mse_loss(self.latent_operate(2, self.latent_operate(0, latent[0])), latent[2])
        loss_bin_vh = F.mse_loss(self.latent_operate(0, self.latent_operate(1, latent[0])), latent[3])
        loss_bin_vv = F.mse_loss(self.latent_operate(1, self.latent_operate(2, latent[0])), latent[0])
        loss_bin_vc = F.mse_loss(self.latent_operate(2, self.latent_operate(1, latent[0])), latent[1])
        loss_bin_ch = F.mse_loss(self.latent_operate(0, self.latent_operate(2, latent[0])), latent[2])
        loss_bin_cv = F.mse_loss(self.latent_operate(1, self.latent_operate(2, latent[0])), latent[1])
        loss_bin_cc = F.mse_loss(self.latent_operate(2, self.latent_operate(2, latent[0])), latent[0])
        loss_bin = loss_bin_hh + loss_bin_hv + loss_bin_hc + loss_bin_vh + loss_bin_vv + loss_bin_vc + loss_bin_ch + loss_bin_cv + loss_bin_cc
        # 3. Total loss
        latent_loss = loss_uni + loss_bin
        # --------------------------------------------------------------------------

        loss = recon_loss + latent_loss * 0.001
        return {"backward": loss, "reconstruction": recon_loss, "latent": latent_loss}

    def forward(self, imgs):
        ''' imgs: [N,3,H,W]
        '''
        imgs_vf = torch.flip(imgs, dims=[2])
        imgs_hf = torch.flip(imgs, dims=[3])
        imgs_cm = torch.flip(imgs, dims=[2,3])

        imgs_bundle = torch.cat([imgs[:,None], imgs_vf[:,None], imgs_hf[:,None], imgs_cm[:,None]], dim=1)
        imgs_batch = einops.rearrange(imgs_bundle, 'b t c h w -> (b t) c h w', t=4)

        latent = self.forward_encoder(imgs_batch)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs_batch, pred, latent)

        pred_0 : torch.Tensor = einops.rearrange(pred, '(b t) l d -> b t l d', t=4).detach()
        pred_0 = pred_0[:,0]

        return loss, self.unpatchify(pred_0)
