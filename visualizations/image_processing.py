import nibabel as nib
import numpy as np

def load_nifti(filepath):
    """
    Función para cargar una imagen NIfTI
    """
    img = nib.load(filepath)  # Cargar imagen
    img_array = img.get_fdata()  # Convertir a NumPy
    return img_array
