import einops
from einops.layers.torch import *
import torch
import torch.nn as nn

if __name__ == '__main__':
    from resnet import *
else:
    from .resnet import *


class PoseResNet(nn.Module):
    def __init__(self, block, layers, # num_joints,
                 predict_mano=True,
                 num_input_channels=3,
                 deconv_with_bias=False,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 final_conv_kernel=1,
                 backbone_ckpt=None,
                 finetune_backbone=True
                 ):
        ''' Pose estimation net using ResNet as backbone.
        backbone_ckpt: dictionary checkpoint or path to pretrained checkpoints.
        '''
        super(PoseResNet, self).__init__()
        self.backbone: ResNet = ResNet(block, layers, num_input_channels,
                                          deconv_with_bias, num_deconv_layers,
                                          num_deconv_filters, num_deconv_kernels,
                                          final_conv_kernel)
        self.hidden_dim = self.backbone.hidden_dim
        self.predict_mano = predict_mano
        if predict_mano:
            self.pose_mlp = nn.Sequential(nn.Linear(self.hidden_dim, 1024),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(1024, 1024),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(1024, 16*3))
        else:
            self.pose_mlp = nn.Sequential(nn.Linear(self.hidden_dim, 1024),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(1024, 1024),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(1024, (21-1)*3))
        self.finetune_backbone = finetune_backbone
        self.backbone.requires_grad_(self.finetune_backbone)

        if isinstance(backbone_ckpt, str):
            backbone_ckpt = torch.load(backbone_ckpt, weights_only=False)
            backbone_ckpt = {k.replace('module.', ''): v for k, v in backbone_ckpt['model'].items()}
            backbone_ckpt = {k: v for k, v in backbone_ckpt.items() if 'deconv_layers' not in k}
        elif isinstance(backbone_ckpt, dict):
            pass
        elif backbone_ckpt is not None:
            raise ValueError('backbone_ckpt must be str or dict')

        if backbone_ckpt is not None:
            missing, unexpected = self.backbone.load_state_dict(backbone_ckpt, strict=False)
            print('missing key(s): ', missing)
            print('unexpected key(s): ', unexpected)
            print(f'Model loaded.')

        if finetune_backbone:  # True
            self.backbone.train()
        else:
            self.backbone.eval()

    def train(self, mode=True):
        self.pose_mlp.train(mode)
        if self.finetune_backbone:
            self.backbone.train(mode)
        else:
            self.backbone.train(False)

    def eval(self):
        self.train(False)

    def extract_feature(self, x):
        feature_maps = self.backbone.forward_featmap(x)
        feats = torch.mean(einops.rearrange(feature_maps, 'b c h w -> b c (h w)'), dim=-1)
        return feats

    def decode_pose(self, feats):
        return einops.rearrange(self.pose_mlp(feats), "b (j d) -> b j d", d=3)

    def forward(self, imgs):
        if self.finetune_backbone:
            feats = self.extract_feature(imgs)
        else:
            with torch.no_grad():
                feats = self.extract_feature(imgs)
        pose = self.decode_pose(feats)
        return pose


if __name__ == "__main__":
    block_class, layers = resnet_spec[50]
    model = PoseResNet(
        block=block_class, layers=layers, num_input_channels=3, backbone_ckpt='/mnt/data_0/renkaiwen/sl_vit/logs/20240816-1/checkpoint-53.pth')
    print(model)

    x = torch.randn(4, 3, 224, 224)
    _, y = model(x)
    print(y.shape)
