import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def read_nifti_file(filepath):
    """Read and load volume"""
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def normalize_scan(volume):
    """Normalize the volume"""
    min_val = np.min(volume)
    max_val = np.max(volume)
    if (max_val - min_val) != 0:
        volume = (volume - min_val) / (max_val - min_val)
    return volume

def zoom_out_and_pad(image, zoom_factor=0.65, output_shape=(160, 160, 128)):
    """
    Aplica zoom out al cerebro y centra la imagen con padding o crop para mantener tamaño fijo.
    """
    zoomed = zoom(image, zoom_factor, order=0)
    zoomed_shape = zoomed.shape

    # Paso 1: Si alguna dimensión es mayor que la deseada → crop
    slices = []
    for i in range(3):
        if zoomed_shape[i] > output_shape[i]:
            start = (zoomed_shape[i] - output_shape[i]) // 2
            end = start + output_shape[i]
            slices.append(slice(start, end))
        else:
            slices.append(slice(0, zoomed_shape[i]))
    zoomed_cropped = zoomed[tuple(slices)]

    # Paso 2: Centrar en canvas
    result = np.zeros(output_shape, dtype=zoomed.dtype)
    insert_slices = []
    for i in range(3):
        start = (output_shape[i] - zoomed_cropped.shape[i]) // 2
        end = start + zoomed_cropped.shape[i]
        insert_slices.append(slice(start, end))

    result[tuple(insert_slices)] = zoomed_cropped
    return result

def resize_volume(img, desired_shape=(128, 128, 96)):
    """Resize across z-axis"""
    current_shape = img.shape
    
    # Padding or cropping para cada dimensión
    for dim in range(3):
        if current_shape[dim] < desired_shape[dim]:
            diff = desired_shape[dim] - current_shape[dim]
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width = [(0, 0)] * 3
            pad_width[dim] = (pad_before, pad_after)
            img = np.pad(img, pad_width, mode='constant')
        else:
            crop_start = (current_shape[dim] - desired_shape[dim]) // 2
            if dim == 0:
                img = img[crop_start:crop_start + desired_shape[dim], :, :]
            elif dim == 1:
                img = img[:, crop_start:crop_start + desired_shape[dim], :]
            else:
                img = img[:, :, crop_start:crop_start + desired_shape[dim]]
        current_shape = img.shape
    
    return img

def center_crop_3d(img, crop_size=(128, 128, 96)):
    """Center crop 3D image"""
    h, w, d = img.shape
    ch, cw, cd = crop_size
    
    x = (h - ch) // 2
    y = (w - cw) // 2
    z = (d - cd) // 2
    
    img_crop = img[x:x+ch, y:y+cw, z:z+cd]
    
    return img_crop

def process_scan(path, for_inference=False):
    """Read and resize volume for training or inference"""
    volume = read_nifti_file(path)
    volume = normalize_scan(volume)
    
    if for_inference:
        volume = zoom_out_and_pad(volume, zoom_factor=0.65)
        volume = resize_volume(volume, desired_shape=(128, 128, 96))
    else:
        volume = resize_volume(volume)
    
    return volume

def process_scany(path, for_inference=False):
    """Read and resize segmentation volume"""
    volume = read_nifti_file(path)
    
    if for_inference:
        volume = zoom_out_and_pad(volume, zoom_factor=0.65)
        volume = resize_volume(volume, desired_shape=(128, 128, 96))
    else:
        volume = resize_volume(volume)
    
    return volume
