from monai.transforms import (
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
    CenterSpatialCropd,
    Compose,
)
from monai.transforms import CropForegroundd
from monai.transforms import ResizeWithPadOrCropd


roi_size = [128, 128, 64]
pixdim = (1.5, 1.5, 2.0)

train_transform = Compose([
    #CropForegroundd(keys=["image", "label"], source_key="image"),
    Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=roi_size),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ToTensord(keys=["image", "label"]),
])

val_transform = Compose([
    #CropForegroundd(keys=["image", "label"], source_key="image"),
    Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=roi_size),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "label"]),
])