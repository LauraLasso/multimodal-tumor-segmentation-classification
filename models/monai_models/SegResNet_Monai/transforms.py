from monai.transforms import (
    Compose, NormalizeIntensityd, RandFlipd, RandRotate90d, Resized, ToTensord
)

target_shape = (224, 224, 144)

train_transform = Compose([
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    Resized(keys=["image", "label"], spatial_size=target_shape, mode=("trilinear", "nearest")),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.5),
    ToTensord(keys=["image", "label"]),
])

val_transform = Compose([
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    Resized(keys=["image", "label"], spatial_size=target_shape, mode=("trilinear", "nearest")),
    ToTensord(keys=["image", "label"]),
])
