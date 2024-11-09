from einops.layers.torch import *
import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

from .ti_vit import TI_ViT


class PoseViT(nn.Module):
    def __init__(self,
                 pose_decoder_depth=4, num_joint_query=16, num_joint_dim=3,
                 # to construct ti_vit
                 img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 # training configuration
                 backbone_ckpt=None, finetune_backbone=True):
        super(PoseViT, self).__init__()
        
        self.num_joint_query = num_joint_query
        
        self.backbone: TI_ViT = TI_ViT(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
            decoder_num_heads= decoder_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss, train_secondary_trans=False)

        # pose decoder
        self.pose_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # contrust query tokens for joints
        self.joint_queries = nn.Parameter(
            torch.empty(size=(1, num_joint_query, decoder_embed_dim), dtype=torch.float32))
        self.joint_queries.requires_grad_(True)
        self.pose_decoder_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                qk_scale=None, norm_layer=norm_layer) for _ in range(pose_decoder_depth)])
        self.pose_decoder_norm = norm_layer(decoder_embed_dim)
        self.pose_decoder_pred = nn.Linear(decoder_embed_dim, num_joint_dim, bias=True)

        self.initialize_weights()
    
    def initialize_weights(self):
        torch.nn.init.normal_(self.joint_queries, std=0.02)
        
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
            
    def forward_backbone(self, imgs: torch.Tensor) -> torch.Tensor:
        '''
        forward feature without masking
        '''
        x = self.backbone.patch_embed(imgs)
        x = x + self.backbone.pos_embed[:, 0:, :]
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        return x

    def forward_pred(self, pred_tokens: torch.Tensor) -> torch.Tensor:
        x = self.pose_decoder_embed(pred_tokens)
        x += self.backbone.decoder_pos_embed
        x = torch.cat([x, self.joint_queries.expand(x.shape[0], -1, -1)], dim=1)
        for blk in self.pose_decoder_blocks:
            x = blk(x)
        x = self.pose_decoder_norm(x[:,-self.num_joint_query:])
        x = self.pose_decoder_pred(x)
        return x

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        patch_tokens: torch.Tensor = self.forward_backbone(imgs)
        cls_tokens: torch.Tensor = torch.mean(patch_tokens, dim=1, keepdim=True)

        pred_tokens: torch.Tensor = torch.cat([cls_tokens, patch_tokens], dim=1)
        
        pose = self.forward_pred(pred_tokens)

        return pose 
