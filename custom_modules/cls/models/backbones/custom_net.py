from mmcls.models.builder import BACKBONES
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcv.cnn import ConvModule, constant_init, kaiming_init


@BACKBONES.register_module()
class CustomNet(BaseBackbone):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        conv_param = dict(
            kernel_size=3,
            stride=2,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            act_cfg=dict(type='ReLU'))

        self.conv1 = ConvModule(3, 16, **conv_param)
        self.conv2 = ConvModule(16, 32, **conv_param)
        self.conv3 = ConvModule(32, 64, **conv_param)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
