import torch
from monai.networks.nets import SegResNet

def get_segresnet_model(device, in_channels=4, out_channels=3):
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=0.2,
    ).to(device)
    return model
