import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
try:
    from utils.common import warnings
except ImportError:
    import warnings
from ..module.dcn import DCN
from ...builder import MODELS

@MODELS.register()
class DCFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 dcn_cfg=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 cfg=None):
        super(DCFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.fpn_convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        # dcn init
        self.dcn_use_lateral = False
        self.dcn_use_augout = False

        if dcn_cfg is not None and dcn_cfg['use_lateral']:
            self.dcn_use_lateral = True
            self.dcn_convs = nn.ModuleList()
            
            for i in range(self.start_level, self.backbone_end_level):
                dcn_conv = DCN(
                    in_channels[i],
                    in_channels[i],
                    kernel_size=(3, 3), 
                    padding=1
                )
                self.dcn_convs.append(dcn_conv)
        if dcn_cfg is not None and dcn_cfg['use_augout']:
            self.dcn_use_augout = True
            self.aug_idx = dcn_cfg['aug_idx']
            self.dcn_conv = DCN(
                out_channels,
                out_channels,
                kernel_size=(3, 3), 
                padding=1
            )


        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) >= len(self.in_channels)
        
        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]
        if self.dcn_use_lateral:
            inputs = [
                dcn_conv(inputs[i + self.start_level])
                for i, dcn_conv in enumerate(self.dcn_convs)
            ]

        # 横向连接 
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # 高层特征上采样与横向连接进行融合，构建自顶向下路径 
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
    
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        if self.dcn_use_augout:
            outs[self.aug_idx] = self.dcn_conv(outs[self.aug_idx])
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
    def dataCheck(self, tensor_lists):
        for i in range(len(tensor_lists)):
            print(i, tensor_lists[i].shape)

if __name__ == '__main__':
    neck_cfg=dict(
        name='DCFPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4,
        dcn_cfg=dict(
            use_lateral = False,
            use_augout = True,
            aug_idx = 1
        )

    )

    # test fpn
    l1 = torch.rand(2, 64, 72, 200)
    l2 = torch.rand(2, 128, 36, 100)
    l3 = torch.rand(2, 256, 18, 50)
    l4 = torch.rand(2, 512, 9, 25)
    inputs = [l1, l2, l3, l4]
    net = MODELS.from_dict(neck_cfg)
    net.to(torch.device('cuda'))
    inputs = [input.to(torch.device('cuda')) for input in inputs]
    output = net(inputs)
    print(net)