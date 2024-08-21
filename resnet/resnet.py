# code from: https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/models/pose_resnet.py
## Reference: https://github.com/microsoft/human-pose-estimation.pytorch

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GlobalAveragePoolingHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)

        batch_size, n_channels = x.shape[:2]
        x = x.view((batch_size, n_channels, -1))
        x = x.mean(dim=-1)

        out = self.head(x)

        return out


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


class ResNet(nn.Module):
    def __init__(self, block, layers, # num_joints,
                 num_input_channels=3,
                 deconv_with_bias=False,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 final_conv_kernel=1,
                 ):
        super().__init__()

        # self.num_joints = num_joints
        self.num_input_channels = num_input_channels
        self.inplanes = 64

        self.deconv_with_bias = deconv_with_bias
        self.num_deconv_layers, self.num_deconv_filters, self.num_deconv_kernels = num_deconv_layers, num_deconv_filters, num_deconv_kernels
        self.final_conv_kernel = final_conv_kernel

        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # get the dimension of latent space 
        self.hidden_dim = self.inplanes

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            self.num_deconv_layers,
            self.num_deconv_filters,
            self.num_deconv_kernels,
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        features = x

        return features


class SLL_ResNet(nn.Module):
    def __init__(self, block, layers, # num_joints,
                 num_input_channels=3,
                 deconv_with_bias=False,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 final_conv_kernel=1,
                 latent_operator_rank=128
                 ):
        super().__init__()

        # ------------------------------------------------------------------------------
        # 1. ResNet backbone 
        self.num_input_channels = num_input_channels
        self.inplanes = 64

        self.deconv_with_bias = deconv_with_bias
        self.num_deconv_layers, self.num_deconv_filters, self.num_deconv_kernels = num_deconv_layers, num_deconv_filters, num_deconv_kernels
        self.final_conv_kernel = final_conv_kernel

        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # get the dimension of hidden space
        self.hidden_dim = self.inplanes

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            self.num_deconv_layers,
            self.num_deconv_filters,
            self.num_deconv_kernels,
        )
        # ------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # 2. SLL operator
        self.vf_operator = nn.Sequential(nn.Linear(self.hidden_dim, latent_operator_rank, bias=False),
                                         nn.Linear(latent_operator_rank, self.hidden_dim, bias=False))
        self.hf_operator = nn.Sequential(nn.Linear(self.hidden_dim, latent_operator_rank, bias=False),
                                         nn.Linear(latent_operator_rank, self.hidden_dim, bias=False))
        self.cm_operator = nn.Sequential(nn.Linear(self.hidden_dim, latent_operator_rank, bias=False),
                                         nn.Linear(latent_operator_rank, self.hidden_dim, bias=False))
        # ------------------------------------------------------------------------------

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            if i < num_layers:
                layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def latent_operate(self, operator_ix, latent):
        ''' latent [N,4,D]
        '''
        if operator_ix == 0:
            latent = self.vf_operator(latent)
        if operator_ix == 1:
            latent = self.hf_operator(latent)
        if operator_ix == 2:
            latent = self.cm_operator(latent)
        return latent

    def forward_encoder(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)
        return feature_map
    
    def forward_decoder(self, x):
        x = self.deconv_layers(x)
        return x

    def forward_loss(self, imgs, pred, latent):
        batch_size = imgs.shape[0]
        # ----------------------------------------------------
        # 1. Reconstruction loss
        imgs_0 = imgs[:,0]
        pred_0 = pred[:,0]
        recon_loss = F.mse_loss(imgs_0.reshape(batch_size, -1), pred_0.reshape(batch_size, -1))
        # ----------------------------------------------------

        # ----------------------------------------------------
        # 2. Structural latent loss
        # (1) Uniary operation
        loss_uni_0 = F.mse_loss(latent[1], self.latent_operate(0, latent[0])) + F.mse_loss(latent[0], self.latent_operate(0, latent[1])) + \
                        F.mse_loss(latent[3], self.latent_operate(0, latent[2])) + F.mse_loss(latent[2], self.latent_operate(0, latent[3]))
        loss_uni_1 = F.mse_loss(latent[2], self.latent_operate(1, latent[0])) + F.mse_loss(latent[0], self.latent_operate(1, latent[2])) + \
                        F.mse_loss(latent[1], self.latent_operate(1, latent[3])) + F.mse_loss(latent[3], self.latent_operate(1, latent[1]))
        loss_uni_2 = F.mse_loss(latent[0], self.latent_operate(2, latent[3])) + F.mse_loss(latent[3], self.latent_operate(2, latent[0])) + \
                        F.mse_loss(latent[1], self.latent_operate(2, latent[2])) + F.mse_loss(latent[2], self.latent_operate(2, latent[1]))
        loss_uni = loss_uni_0 + loss_uni_1 + loss_uni_2
        # (2) Binary operation
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
        # (3) Total loss
        latent_loss = loss_uni + loss_bin
        # ----------------------------------------------------

        loss = recon_loss + 1e-3 * latent_loss
        return {'backward': loss, 'reconstruction': recon_loss, 'latent': latent_loss}

    def forward(self, imgs):
        ''' imgs: [N,3,H,W]
        '''
        imgs_vf = torch.flip(imgs, dims=[2])
        imgs_hf = torch.flip(imgs, dims=[3])
        imgs_cm = torch.flip(imgs, dims=[2,3])

        imgs_bundle = torch.cat([imgs[:,None], imgs_vf[:,None], imgs_hf[:,None], imgs_cm[:,None]], dim=1)
        imgs_batch = einops.rearrange(imgs_bundle, 'b t c h w -> (b t) c h w', t=4)

        feature_map = self.forward_encoder(imgs_batch)
        pred = self.forward_decoder(feature_map)

        imgs_batch = einops.rearrange(imgs_batch, '(b t) c h w -> b t c h w', t=4)
        pred = einops.rearrange(pred, '(b t) c h w -> b t c h w', t=4)
        feature_map = einops.rearrange(feature_map, '(b t) c h w -> b t c h w', t=4)
        latent = torch.mean(einops.rearrange(feature_map, 'b t c h w -> t b c (h w)', t=4), dim=-1)

        loss = self.forward_loss(imgs_batch, pred, latent)

        pred_0 = pred[:,0].detach()

        return loss, pred_0
