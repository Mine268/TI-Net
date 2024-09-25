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
                 num_input_channels=3,
                 deconv_with_bias=False,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 final_conv_kernel=1,
                 backbone_ckpt=None
                 ):
        ''' Pose estimation net using ResNet as backbone.
        backbone_ckpt: dictionary checkpoint or path to pretrained checkpoints.
        '''
        super(PoseResNet, self).__init__()
        self.backbone: nn.Module = ResNet(block, layers, num_input_channels,
                                          deconv_with_bias, num_deconv_layers,
                                          num_deconv_filters, num_deconv_kernels,
                                          final_conv_kernel)
        self.hidden_dim = self.backbone.hidden_dim
        self.pose_mlp = nn.Sequential(nn.Linear(self.hidden_dim, 1024),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(inplace=True),
                                      # nn.Linear(1024, 1024),
                                      # nn.ReLU(inplace=True),
                                      # nn.Linear(1024, 1024),
                                      # nn.ReLU(inplace=True),
                                      nn.Linear(1024, 16*3),
                                      Rearrange('b (j d) -> b j d', j=16, d=3))
        self.pretrained_backbone = backbone_ckpt is not None
        self.backbone.requires_grad_(self.pretrained_backbone)

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
            self.backbone.eval()
        else:
            self.backbone.train()

    def train(self, mode: bool = True):
        if not self.pretrained_backbone:
            self.backbone.train(mode)
        self.pose_mlp.train(mode)
        print(f"trigger TRAIN mode, train backbone: {(not self.pretrained_backbone) and mode}")
        return self

    def eval(self):
        if not self.pretrained_backbone:
            self.backbone.eval()
        self.pose_mlp.eval()
        print(f"trigger EVAL mode")
        return self

    def extract_feature(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        feature_maps = self.backbone.layer4(x)
        feats = torch.mean(einops.rearrange(feature_maps, 'b c h w -> b c (h w)'), dim=-1)
        return feats

    def decode_pose(self, feats):
        return self.pose_mlp(feats)

    def forward(self, imgs):
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
