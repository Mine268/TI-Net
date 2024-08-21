from .resnet import *


class PoseResNet(nn.Module):
    def __init__(self, block, layers, # num_joints,
                 num_input_channels=3,
                 deconv_with_bias=False,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 final_conv_kernel=1,
                 ):
        pass
