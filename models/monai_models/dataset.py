import os
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset


def get_augmentations(phase):
    from monai.transforms import Compose
    list_transforms = []
    list_trfms = Compose(list_transforms)
    return list_trfms

class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str="test", is_resize: bool=False, transform=None):
        self.df = df
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        self.is_resize = is_resize
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        try:
            id_ = self.df.loc[idx, 'Brats20ID']
            root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
            images = []
            for data_type in self.data_types:
                img_path = os.path.join(root_path, id_ + data_type)
                img = self.load_img(img_path)
                if self.is_resize:
                    img = self.resize(img)
                img = self.normalize(img)
                images.append(img)
            img = np.stack(images)
            sample = {"image": img}
            mask_path = os.path.join(root_path, id_ + "_seg.nii")
            mask = self.load_img(mask_path)
            if self.is_resize:
                mask = self.resize(mask)
            mask = self.preprocess_mask_labels(mask)
            sample["label"] = mask
            if self.augmentations:
                if img.shape[1:] == mask.shape[1:]:
                    augmented = self.augmentations(
                        image=img.astype(np.float32),
                        mask=mask.astype(np.float32)
                    )
                    img = augmented['image']
                    mask = augmented['mask']
                else:
                    print(f"Shapes no coinciden en sample {idx}: img {img.shape}, mask {mask.shape}")
            sample = {
                "Id": id_,
                "image": img,
                "label": mask,
            }
            if self.transform:
                sample = self.transform(sample)
            return sample
        except Exception as e:
            print(f"Fallo en sample {idx}: {e}")
            raise e

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, img: np.ndarray):
        from skimage.transform import resize
        return resize(img, (128, 128, 128), preserve_range=True)

    def preprocess_mask_labels(self, mask: np.ndarray):
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1
        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1
        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1
        mask = np.stack([mask_WT, mask_TC, mask_ET])
        return mask
