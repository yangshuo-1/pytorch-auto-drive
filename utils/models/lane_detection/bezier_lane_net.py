import torch
import torch.nn as nn
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast
else:
    from utils.torch_amp_dummy import autocast

from .bezier_base import BezierBaseNet
from ..builder import MODELS


@MODELS.register()
class BezierLaneNet(BezierBaseNet):
    # Curve regression network, similar design as simple object detection (e.g. FCOS)
    def __init__(self,
                 backbone_cfg,
                 reducer_cfg,
                 dilated_blocks_cfg,
                 feature_fusion_cfg,
                 head_cfg,
                 aux_seg_head_cfg,
                 image_height=360,
                 num_regression_parameters=8,
                 thresh=0.5,
                 local_maximum_window_size=9):
        super(BezierLaneNet, self).__init__(thresh, local_maximum_window_size)
        global_stride = 16          # 空间下采样比例 
        branch_channels = 256       # 特征通道数 

        self.backbone = MODELS.from_dict(backbone_cfg)
        self.reducer = MODELS.from_dict(reducer_cfg)
        self.dilated_blocks = MODELS.from_dict(dilated_blocks_cfg)
        self.simple_flip_2d = MODELS.from_dict(feature_fusion_cfg)  # Name kept for legacy weights FeatureFlipFusion
        self.aggregator = nn.AvgPool2d(kernel_size=((image_height - 1) // global_stride + 1, 1), stride=1, padding=0)
        self.regression_head = MODELS.from_dict(head_cfg)  # Name kept for legacy weights
        self.proj_classification = nn.Conv1d(branch_channels, 1, kernel_size=1, bias=True, padding=0)
        self.proj_regression = nn.Conv1d(branch_channels, num_regression_parameters,
                                         kernel_size=1, bias=True, padding=0)
        self.segmentation_head = MODELS.from_dict(aux_seg_head_cfg)

    def forward(self, x):
        # Return shape: B x Q, B x Q x N x 2
        # 1. input输入：[batch size, 3, 288, 800]
        x = self.backbone(x)                                    # 2. backbone输出：[batch size, 256, 18, 50]
        if isinstance(x, dict):
            x = x['out']

        if self.reducer is not None:
            x = self.reducer(x)

        # Segmentation task
        # 辅助的分割任务，推理时不用 
        if self.segmentation_head is not None:
            segmentations = self.segmentation_head(x)           # 用backbone输出的特征进行分割，输出：[batch size, 1, 18, 50]
        else:
            segmentations = None

        # 重构了ResNet的扩张模块 
        if self.dilated_blocks is not None:
            x = self.dilated_blocks(x)                          # 3. 扩张模块输出： [batch size, 256, 18, 50]

        # 特征翻转模块 
        with autocast(False):  # TODO: Support fp16 like mmcv
            x = self.simple_flip_2d(x.float())                  # 4. 特征反转模块输出：[batch size, 256, 18, 50]
       
        
        x = self.aggregator(x)[:, :, 0, :]                      # 5. 结果：[batch size, 256, 50] aggregator输出[batch size, 1, 256, 50],高度方向上求平均值 

        x = self.regression_head(x)                             # 6. regression_head[batch size, 256, 50]
        logits = self.proj_classification(x).squeeze(1)         # 7. classification输出 [batch size, 50] 候选 
        curves = self.proj_regression(x)                        # 8. regression输出 [batch size, 8, 50]

        return {'logits': logits,
                'curves': curves.permute(0, 2, 1).reshape(curves.shape[0], -1, curves.shape[-2] // 2, 2).contiguous(),
                'segmentations': segmentations}
    
    # 评估模式 
    def eval(self, profiling=False):
        # 将模型设置为评估模式 
        super().eval()
        # 评估模式时关闭分割头 
        if profiling:
            self.segmentation_head = None
        return self
