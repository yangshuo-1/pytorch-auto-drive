from utils.models import MODELS


if __name__ == '__main__':
    neck_cfg=dict(
        name='CBAFPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4
    )
    net = MODELS.from_dict(neck_cfg)
    print(net)