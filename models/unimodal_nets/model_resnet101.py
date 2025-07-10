import segmentation_models_pytorch as smp

def get_resnet101_unet(in_channels=3, classes=1):
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes
    )
    return model
