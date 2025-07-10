import segmentation_models_pytorch as smp

def get_resnet34_unet(in_channels=3, classes=1):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes
    )
    return model
