import nibabel as nib
import numpy as np
import torch
import cv2

def load_nifti_as_2d_rgb(filepath, slice_axis=2, slice_index=None, target_size=(1024, 1024)):
    nii_img = nib.load(filepath)
    img_data = nii_img.get_fdata()
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
    if slice_index is None:
        slice_index = img_data.shape[slice_axis] // 2
    if slice_axis == 0:
        img_2d = img_data[slice_index, :, :]
    elif slice_axis == 1:
        img_2d = img_data[:, slice_index, :]
    else:
        img_2d = img_data[:, :, slice_index]
    img_2d = cv2.resize(img_2d, target_size)
    img_rgb = np.stack([img_2d] * 3, axis=0)
    img_tensor = torch.tensor(img_rgb, dtype=torch.float32).unsqueeze(0)
    return img_tensor.cuda()

def infer(model, image_tensor):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        results = model(image_tensor)
    return results
