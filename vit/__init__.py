from functools import partial

import torch.nn as nn
from .vit import VisionTransformer, VitAutoEncoder, SL_VitAutoEncoder, SLL_VitAutoEncoder
from .mae import MaskedAutoencoderViT


# -----------------
# vit

# vit
def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# ae vit
def ae_vit_base_patch16_dec512d8b(**kwargs):
    model = VitAutoEncoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def ae_vit_large_patch16_dec512d8b(**kwargs):
    model = VitAutoEncoder(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def ae_vit_huge_patch14_dec512d8b(**kwargs):
    model = VitAutoEncoder(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

ae_vit_base_patch16 = ae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
ae_vit_large_patch16 = ae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
ae_vit_huge_patch14 = ae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


# sl_vit
def sl_vit_base_patch16_dec512d8b(**kwargs):
    model = SL_VitAutoEncoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def sl_vit_large_patch16_dec512d8b(**kwargs):
    model = SL_VitAutoEncoder(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def sl_vit_huge_patch14_dec512d8b(**kwargs):
    model = SL_VitAutoEncoder(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

sl_vit_base_patch16 = sl_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
sl_vit_large_patch16 = sl_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
sl_vit_huge_patch14 = sl_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


# sll_vit
def ssl_vit_base_patch16_dec512d8b(**kwargs):
    model = SLL_VitAutoEncoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def ssl_vit_large_patch16_dec512d8b(**kwargs):
    model = SLL_VitAutoEncoder(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def ssl_vit_huge_patch14_dec512d8b(**kwargs):
    model = SLL_VitAutoEncoder(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

ssl_vit_base_patch16 = ssl_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
ssl_vit_large_patch16 = ssl_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
ssl_vit_huge_patch14 = ssl_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


# ----------------------------
# mae

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
