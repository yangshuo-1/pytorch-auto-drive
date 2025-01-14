import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
try:
    from utils.common import warnings
except ImportError:
    import warnings
from utils.models.common_models.module.dcn import AUX_DCN
from ...builder import MODELS

class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()
        # 原来是封装了conv和bn，现在拆开
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False)
        # self.norm_atten = nn.BatchNorm2d(in_chan)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        # weight_init.c2_xavier_fill(self.conv_atten)
        # weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        identity = x
        x = F.avg_pool2d(x, x.size()[2:])   # 通道注意力 
        x = self.conv_atten(x)
                
        atten = self.sigmoid(x)
        feat = torch.mul(x, atten)
        x = identity + feat
        feat = self.conv(x)
        return feat


class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc)
        self.offset = nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False)
        self.dcpack_L2 = AUX_DCN(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deform_groups=8)
        self.relu = nn.ReLU(inplace=True)
        # weight_init.c2_xavier_fill(self.offset)

    def forward(self, feat_l, feat_s, main_path=None):
        # feat_s: 高层的特征  feat_l: 底层特征
        # 下采样
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        # 横向连接 FSM 
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        # 计算offset  这里只学习offset，回头试试加上mask
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif 通道上拼接 
        feat_align = self.relu(self.dcpack_L2(feat_up, offset))  # [feat, offset]
        return feat_align + feat_arm

@MODELS.register()
class FAFPN(nn.Module):
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
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 cfg=None):
        super(FAFPN, self).__init__()
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

        align_modules = nn.ModuleList()
        fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level - 1):
            align_module = FeatureAlign_V2(
                in_nc=in_channels[i],
                out_nc=out_channels
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            align_modules.append(align_module)
            fpn_convs.append(fpn_conv)
        # 最顶层的横向连接 
        top_conv = ConvModule(
                in_channels[-1],
                out_channels,
                1,
                inplace=False)
        align_modules.append(top_conv)
        
        self.align_modules = align_modules[::-1]
        self.fpn_convs = fpn_convs[::-1]
        

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs: feature list from backbone. order should be from high to low resolution.
            outputs: feature list after process. from high to low resolution
        """
        assert len(inputs) == len(self.in_channels)
        # 顺序反转，从低分辨率到高分辨率 
        inputs = inputs[::-1]
        # used_backbone_levels = len(inputs)
        result = []
        pre_feature = self.align_modules[0](inputs[0])      # 顶层处理
        result.append(pre_feature)
        # 除了顶层之外的处理 
        for feature, aliign, out_conv in zip(inputs[1:], self.align_modules[1:], self.fpn_convs[0:]):
            pre_feature = aliign(feature, pre_feature)
            pre_feature = out_conv(pre_feature)
            result.insert(0, pre_feature)
    
        # build outputs
        # part 1: from original levels
        # outs = [
        #     self.fpn_convs[i](result[i]) for i in range(used_backbone_levels)
        # ]
        return tuple(result)
    
    def dataCheck(self, tensor_lists):
        for i in range(len(tensor_lists)):
            print(i, tensor_lists[i].shape)

if __name__ == '__main__':
    neck_cfg=dict(
        name='FAFPN',
        in_channels=[128, 256, 512],
        out_channels=256,
        num_outs=3
    )

    # l1 = torch.rand(2, 64, 72, 200)
    l2 = torch.rand(2, 128, 36, 100)
    l3 = torch.rand(2, 256, 18, 50)
    l4 = torch.rand(2, 512, 9, 25)
    inputs = [l2, l3, l4]
    net = MODELS.from_dict(neck_cfg)
    net.to(torch.device('cuda'))
    inputs = [input.to(torch.device('cuda')) for input in inputs]
    output = net(inputs)
    print(net)