# En este código se cargan las imágenes junto con un punto cada una.
import os
import numpy as np
import cv2
import torch
import nibabel as nib
from torch.utils.data import Dataset

def load_nifti(filepath):
    nii_img = nib.load(filepath)
    img_data = nii_img.get_fdata(dtype=np.float32)
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data) + 1e-6)
    return img_data

def combine_masks(mask_paths):
    combined_mask = None
    for mask_path in mask_paths:
        mask = load_nifti(mask_path).astype(np.float32)
        if combined_mask is None:
            combined_mask = mask
        else:
            np.maximum(combined_mask, mask, out=combined_mask)
    return combined_mask

# Esta función coge un punto que caiga dentro de la máscara (1)
def random_click(mask, point_label=1):
    max_label = max(set(mask.flatten()))
    if round(max_label) == 0:
        point_label = round(max_label)
    indices = np.argwhere(mask == max_label)
    return point_label, indices[np.random.randint(len(indices))]

class NIfTIDataset(Dataset):
    def __init__(self, patients, dataset_path, max_patients, image_size=1024, mask_size=1024, slice_axis=2):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.mask_size = mask_size
        self.slice_axis = slice_axis
        self.patients = []
        self.max_patients = max_patients

        valid_count = 0
        for patient_id in patients:
            patient_path = os.path.join(self.dataset_path, patient_id)
            files = sorted(os.listdir(patient_path))

            mask_files = [os.path.join(patient_path, f) for f in files if "mask" in f]
            if not mask_files:
                continue

            combined_mask = combine_masks(mask_files)
            mask_slice = combined_mask[:, :, combined_mask.shape[2] // 2]
            mask_resized = cv2.resize(mask_slice, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
            mask_binarized = (mask_resized > 0.5).astype(np.uint8)

            _, pt = random_click(mask_binarized)
            if pt is not None and len(pt) == 2:
                self.patients.append(patient_id)
                valid_count += 1

            if valid_count >= self.max_patients:
                break

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        patient_path = os.path.join(self.dataset_path, patient_id)
        files = sorted(os.listdir(patient_path))

        image_file = next((f for f in files if "image" in f), None)
        image_path = os.path.join(patient_path, image_file)
        img_data = load_nifti(image_path)
        img_slice = img_data[:, :, img_data.shape[2] // 2]
        img_resized = cv2.resize(img_slice, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        #img_rgb = np.stack([img_resized] * 3, axis=0)  # [3, H, W]
        #img_tensor = torch.tensor(img_rgb, dtype=torch.float32)

        img_rgb = np.stack([img_resized] * 3, axis=-1)  # [H, W, 3]
        img_rgb = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        img_rgb = img_rgb.transpose(2, 0, 1)  # [3, H, W] para PyTorch
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32)

        mask_files = [os.path.join(patient_path, f) for f in files if "mask" in f]
        combined_mask = combine_masks(mask_files)
        mask_slice = combined_mask[:, :, combined_mask.shape[2] // 2]
        mask_resized = cv2.resize(mask_slice, (self.mask_size, self.mask_size), interpolation=cv2.INTER_NEAREST)
        mask_binarized = (mask_resized > 0.5).astype(np.uint8)

        point_label, pt = random_click(mask_binarized)
        point_tensor_coords = torch.tensor(pt, dtype=torch.float32).unsqueeze(0)        # [1, 2]
        point_tensor_labels = torch.tensor(point_label, dtype=torch.int64)  # [1, 1]

        mask_tensor_binarized = torch.tensor(mask_binarized, dtype=torch.float32).unsqueeze(0)
        mask_tensor_original = torch.tensor(mask_resized, dtype=torch.float32).unsqueeze(0)

        image_meta_dict = {'filename_or_obj': patient_id}

        return {
            'image': img_tensor,  # [3, image_size, image_size]
            'multi_rater': mask_tensor_original,  # [1, mask_size, mask_size]
            'p_label': point_tensor_labels,       # [1, 1]
            'pt': point_tensor_coords,            # [1, 2]
            'mask': mask_tensor_binarized,        # [1, mask_size, mask_size]
            'mask_ori': mask_tensor_original,     # same as multi_rater
            'image_meta_dict': image_meta_dict
        }