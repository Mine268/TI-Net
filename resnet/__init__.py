from .resnet import ResNet, resnet_spec
from .resnet import SLL_ResNet, SL4_ResNet
from .posenet import PoseResNet


def pose_resnet18(backbone_ckpt=None, predict_mano=True, finetune_backbone=True):
    block_class, layers = resnet_spec[18]
    model = PoseResNet(predict_mano=predict_mano,
                       block=block_class,
                       layers=layers,
                       num_input_channels=3,
                       backbone_ckpt=backbone_ckpt,
                       finetune_backbone=finetune_backbone)
    return model

def pose_resnet34(backbone_ckpt=None, predict_mano=True, finetune_backbone=True):
    block_class, layers = resnet_spec[34]
    model = PoseResNet(predict_mano=predict_mano,
                       block=block_class,
                       layers=layers,
                       num_input_channels=3,
                       backbone_ckpt=backbone_ckpt,
                       finetune_backbone=finetune_backbone)
    return model

def pose_resnet50(backbone_ckpt=None, predict_mano=True, finetune_backbone=True):
    block_class, layers = resnet_spec[50]
    model = PoseResNet(predict_mano=predict_mano,
                       block=block_class,
                       layers=layers,
                       num_input_channels=3,
                       backbone_ckpt=backbone_ckpt,
                       finetune_backbone=finetune_backbone)
    return model

def pose_resnet101(backbone_ckpt=None, predict_mano=True, finetune_backbone=True):
    block_class, layers = resnet_spec[101]
    model = PoseResNet(predict_mano=predict_mano,
                       block=block_class,
                       layers=layers,
                       num_input_channels=3,
                       backbone_ckpt=backbone_ckpt,
                       finetune_backbone=finetune_backbone)
    return model

def pose_resnet152(backbone_ckpt=None, predict_mano=True, finetune_backbone=True):
    block_class, layers = resnet_spec[152]
    model = PoseResNet(predict_mano=predict_mano,
                       block=block_class,
                       layers=layers,
                       num_input_channels=3,
                       backbone_ckpt=backbone_ckpt,
                       finetune_backbone=finetune_backbone)
    return model


def sll_resnet18(**kwargs):
    block_class, layers = resnet_spec[18]
    model = SLL_ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1
    )
    return model

def sll_resnet34(**kwargs):
    block_class, layers = resnet_spec[34]
    model = SLL_ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1
    )
    return model

def sll_resnet50(**kwargs):
    block_class, layers = resnet_spec[50]
    model = SLL_ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1
    )
    return model

def sll_resnet101(**kwargs):
    block_class, layers = resnet_spec[101]
    model = SLL_ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1
    )
    return model

def sll_resnet152(**kwargs):
    block_class, layers = resnet_spec[152]
    model = SLL_ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1
    )
    return model



def sl4_resnet18(**kwargs):
    block_class, layers = resnet_spec[18]
    model = SL4_ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1,
        **kwargs
    )
    return model

def sl4_resnet34(**kwargs):
    block_class, layers = resnet_spec[34]
    model = SL4_ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1,
        **kwargs
    )
    return model

def sl4_resnet50(**kwargs):
    block_class, layers = resnet_spec[50]
    model = SL4_ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1,
        **kwargs
    )
    return model

def sl4_resnet101(**kwargs):
    block_class, layers = resnet_spec[101]
    model = SL4_ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1,
        **kwargs
    )
    return model

def sl4_resnet152(**kwargs):
    block_class, layers = resnet_spec[152]
    model = SL4_ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1,
        **kwargs
    )
    return model


def resnet18():
    block_class, layers = resnet_spec[18]
    model = ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1
    )
    return model

def resnet34():
    block_class, layers = resnet_spec[34]
    model = ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1
    )
    return model

def resnet50():
    block_class, layers = resnet_spec[50]
    model = ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1
    )
    return model

def resnet101():
    block_class, layers = resnet_spec[101]
    model = ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1
    )
    return model

def resnet152():
    block_class, layers = resnet_spec[152]
    model = ResNet(
        block=block_class, layers=layers,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=5,
        num_deconv_filters=[256, 256, 256, 256, 3],
        num_deconv_kernels=[4, 4, 4, 4, 4],
        final_conv_kernel=1
    )
    return model
