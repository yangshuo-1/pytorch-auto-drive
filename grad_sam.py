import argparse
import torch
import cv2
import numpy as np
from importmagician import import_from
with import_from('./'):
    from utils.args import read_config, parse_arg_cfg, cmd_dict, add_shortcuts
    from utils.runners import LaneDetDir
    try:
        from utils.common import warnings
    except ImportError:
        import warnings
from utils.models.lane_detection import BezierLaneNet
from utils.common import load_checkpoint
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from torchvision import models
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from torchvision import transforms
from PIL import Image
import torchvision
from utils.transforms import TRANSFORMS
from utils.models import MODELS


class BezierNetOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(BezierNetOutputWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['logits']

def GradSAM(cfg):
    model = MODELS.from_dict(cfg['model'])
   
    ckpt_filename = cfg['test']['checkpoint']
    load_checkpoint(net=model, lr_scheduler=None, optimizer=None, filename=ckpt_filename)
    model = BezierNetOutputWrapper(model)
    model.to(torch.device('cuda'))

    transforms=TRANSFORMS.from_dict(cfg['test_augmentation'])

    img_path = '/home/yang/project/autodrive/lane_detection/pytorch-auto-drive/test_culane_vis/driver_100_30frame/05251624_0451.MP4/03480.jpg'
    img = Image.open(img_path).convert('RGB')
    img_tensor = transforms(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    img_tensor = img_tensor.to(torch.device('cuda'))

    target_layers = [model.model.backbone.layer3[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    
    grayscale_cam = cam(input_tensor=img_tensor)
    # 取第1张图的cam
    grayscale_cam = grayscale_cam[0, :]
    # 将CAM作为掩码(mask)叠加到原图上
    img = img.resize((800,288))
    img_array = np.array(img).astype(np.float32)/255.0
    
    cam_image = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    plt.imshow(cam_image)
    plt.show()


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PytorchAutoDrive lane directory vis', conflict_handler='resolve')
    add_shortcuts(parser)

    parser.add_argument('--config', type=str, help='Path to config file', required=True)
    parser.add_argument('--image-path', type=str,
                        help='Image input path')
    parser.add_argument('--save-path', type=str,
                        help='Result output path')

    # Optional args/to overwrite configs
    parser.add_argument('--mask-path', type=str,
                        help='Mask input path, if both mask & keypoint are None,'
                             'inference will be performed ')
    parser.add_argument('--keypoint-path', type=str,
                        help='Keypoint input path (expect json/txt file in CULane format, [x, y]),'
                             'if both mask & keypoint are None, inference will be performed')
    parser.add_argument('--gt-keypoint-path', type=str,
                        help='Ground truth keypoint input path (expect json/txt file in CULane format, [x, y]),'
                             'if both mask & keypoint are None, inference will be performed')
    # 各种后缀的配置 
    parser.add_argument('--image-suffix', type=str, default='.jpg',
                        help='Image file suffix')
    parser.add_argument('--keypoint-suffix', type=str, default='.lines.txt',
                        help='Keypoint file suffix')
    parser.add_argument('--gt-keypoint-suffix', type=str, default='.lines.txt',
                        help='Ground truth keypoint file suffix')
    parser.add_argument('--mask-suffix', type=str, default='.png',
                        help='Segmentation mask file suffix')
    
    parser.add_argument('--style', type=str, default='point',
                        help='Lane visualization style: point/line/bezier')
    parser.add_argument('--metric', type=str, default='culane',
                        help='Lane eval metric when comparing with GT')
    parser.add_argument('--pred', action='store_true',
                        help='Whether to predict from a model')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--use-color-pool', action='store_true',
                        help='Use a larger color pool for lane segmentation masks')
    parser.add_argument('--cfg-options', type=cmd_dict,
                        help='Override config options with \"x1=y1 x2=y2 xn=yn\"')
    # 互斥参数组 
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--continue-from', type=str,
                        help='[Deprecated] Continue training from a previous checkpoint')
    group2.add_argument('--checkpoint', type=str,
                        help='Continue/Load from a previous checkpoint')

    retain_args = ['mixed_precision', 'pred', 'metric',
                   'image_path', 'save_path', 'mask_path', 'keypoint_path', 'gt_keypoint_path',
                   'image_suffix', 'keypoint_suffix', 'gt_keypoint_suffix', 'mask_suffix', 'use_color_pool', 'style']

    args = parser.parse_args()

    # Parse configs and build model
    if args.mixed_precision and torch.__version__ < '1.6.0':
        warnings.warn('PyTorch version too low, mixed precision training is not available.')
    if args.image_path is not None and args.save_path is not None:
        assert args.image_path != args.save_path, "Try not to overwrite your dataset!"
    cfg = read_config(args.config)
    args, cfg = parse_arg_cfg(args, cfg)

    cfg_runner_key = 'vis' if 'vis' in cfg.keys() else 'test'           # 是否是vis模式
    for k in retain_args:
        cfg[cfg_runner_key][k] = vars(args)[k]

    # bezier必须要预测 
    if not cfg[cfg_runner_key]['pred']:
        assert cfg[cfg_runner_key]['style'] != 'bezier', 'Must use --pred for style bezier!'
        cfg['model'] = None
    
    GradSAM(cfg)
    

