from importmagician import import_from
with import_from('./'):
    # Data pipeline
    from configs.lane_detection.common.datasets.culane_bezier import dataset
    from configs.lane_detection.common.datasets.train_level1b_288 import train_augmentation
    from configs.lane_detection.common.datasets.test_288 import test_augmentation

    # Optimization pipeline
    from configs.lane_detection.common.optims.matchingloss_bezier import loss
    from configs.lane_detection.common.optims.adam00006_dcn import optimizer
    from configs.lane_detection.common.optims.ep36_cosine import lr_scheduler

    # Define vis_dataset
    from configs.lane_detection.common.datasets._utils import CULANE_ROOT


train = dict(
    exp_name='resnet18_bezierlanenet_culane-aug2-bs32',
    workers=4,
    batch_size=32,
    checkpoint=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda',

    val_num_steps=0,  # Seg IoU validation (mostly useless)
    save_dir='./checkpoints',

    input_size=(288, 800),
    original_size=(590, 1640),
    num_classes=None,
    num_epochs=36,
    collate_fn='dict_collate_fn',  # 'dict_collate_fn' for LSTR
    seg=False,  # Seg-based method or not
)

test = dict(
    exp_name='resnet18_bezierlanenet_culane-aug2-bs32',
    workers=0,
    batch_size=1,
    checkpoint='./checkpoints/resnet18_bezierlanenet_culane-aug2-bs32/model.pt',
    # Device args
    device='cuda',

    save_dir='./checkpoints',

    seg=False,
    gap=20,
    ppl=18,
    thresh=None,
    collate_fn='dict_collate_fn',  # 'dict_collate_fn' for LSTR
    input_size=(288, 800),
    original_size=(590, 1640),
    max_lane=4,
    dataset_name='culane'
)

model = dict(
    name='MyModel',
    image_height=288,
    num_regression_parameters=8,  # 3 x 2 + 2 = 8 (Cubic Bezier Curve)

    # Inference parameters
    thresh=0.95,
    local_maximum_window_size=9,

    # Backbone (3-stage resnet (no dilation) + 2 extra dilated blocks)
    backbone_cfg=dict(
        name='predefined_resnet_backbone',
        backbone_name='resnet18',
        return_layer={
            'layer1': 'l1',
            'layer2': 'l2',
            'layer3': 'l3',
            'layer4': 'l4'
        },
        pretrained=True,
        replace_stride_with_dilation=[False, False, False]
    ),
    reducer_cfg=None,  # No need here
    dilated_blocks_cfg=dict(
        name='predefined_dilated_blocks',
        in_channels=256,
        mid_channels=64,
        dilations=[4, 8]
    ),

    # Head, Fusion module
    feature_fusion_cfg=dict(
        name='FeatureFlipFusion',
        channels=256
    ),
    head_cfg=dict(
        name='ConvProjection_1D',
        num_layers=2,
        in_channels=256,
        bias=True,
        k=3
    ),  # Just some transforms of feature, similar to FCOS heads, but shared between cls & reg branches

    # Auxiliary binary segmentation head (automatically discarded in eval() mode)
    aux_seg_head_cfg=dict(
        name='SimpleSegHead',
        in_channels=256,
        mid_channels=64,
        num_classes=1
    ),
    neck_cfg=dict(
        name='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4
    )

)


vis_dataset = dict(
    name='CULaneVis',
    root_dataset=CULANE_ROOT,
    root_output='./test_culane_vis',
    root_keypoint=None,
    image_set='check'
)