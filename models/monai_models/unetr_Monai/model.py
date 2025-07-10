import torch
from monai.networks.nets import UNETR as UNETR_monai

def get_unetr_model(device, in_channels=4, out_channels=3, img_size=(224,224,144), roi_size = [128, 128, 64], embed_dim= 768):
    model = UNETR_monai(
        in_channels=4,
        out_channels=3,
        img_size=tuple(roi_size),
        feature_size=16,
        hidden_size=embed_dim,
        mlp_dim=3072,
        num_heads=12,
        #pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)
    return model
