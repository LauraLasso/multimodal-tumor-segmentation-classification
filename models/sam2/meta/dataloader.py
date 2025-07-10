# En este código se cargan las imágenes junto con 10 puntos cada una.
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from models.sam2.meta.utils import *
class NIfTIDataset(Dataset):
    def __init__(self, patients, dataset_path, max_patients, target_size=(512, 512), slice_axis=2):
        self.dataset_path = dataset_path
        self.target_size = target_size
        self.slice_axis = slice_axis
        self.patients = []
        self.max_patients = max_patients

        # **No carga imágenes en memoria**
        valid_count = 0
        for patient_id in patients:
            patient_path = os.path.join(self.dataset_path, patient_id)
            files = sorted(os.listdir(patient_path))

            # Verifica si tiene máscaras sin cargarlas
            #mask_files = [os.path.join(patient_path, f) for f in files if "mask" in f]
            mask_files = [os.path.join(patient_path, f) for f in files 
              if "mask" in f and f.endswith((".nii", ".nii.gz")) and "(1)" not in f]
            if not mask_files:
                continue

            # Fusiona máscaras sin cargarlas completamente en RAM
            combined_mask = combine_masks(mask_files)
            mask_binarized = cv2.resize(combined_mask[:, :, combined_mask.shape[2] // 2], self.target_size, interpolation=cv2.INTER_NEAREST)
            mask_binarized = (mask_binarized > 0.5).astype(np.uint8)

            # Verifica si hay puntos clave
            point_coords, _ = extract_keypoints(mask_binarized, num_points=10)
            if point_coords.shape[0] > 0:
                self.patients.append(patient_id)  # Solo guarda IDs
                valid_count += 1
                print(valid_count)

            del combined_mask, mask_binarized  # Liberar memoria intermedia
            torch.cuda.empty_cache()

            if valid_count >= self.max_patients:
                break

        print(f"Pacientes válidos para entrenamiento: {len(self.patients)}")

    def resize_image(self, img):
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

    def resize_mask(self, mask):
        return cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

    def extract_slice(self, img_data):
        slice_index = img_data.shape[self.slice_axis] // 2
        return img_data[:, :, slice_index]

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        patient_path = os.path.join(self.dataset_path, patient_id)
        files = sorted(os.listdir(patient_path))

        # Cargar imagen
        image_file = next((f for f in files if "image" in f), None)
        if not image_file:
            return None

        image_path = os.path.join(patient_path, image_file)
        img_data = load_nifti(image_path)
        img_2d = cv2.resize(img_data[:, :, img_data.shape[2] // 2], self.target_size, interpolation=cv2.INTER_LINEAR)
        img_rgb = np.stack([img_2d] * 3, axis=0)  # Convertir a RGB
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).unsqueeze(0)

        del img_data, img_2d, img_rgb
        torch.cuda.empty_cache()

        # Cargar máscaras
        #mask_files = [os.path.join(patient_path, f) for f in files if "mask" in f]
        mask_files = [os.path.join(patient_path, f) for f in files if f.endswith(('.nii', '.nii.gz'))]
        combined_mask = combine_masks(mask_files)

        mask_original = cv2.resize(combined_mask[:, :, combined_mask.shape[2] // 2], self.target_size, interpolation=cv2.INTER_NEAREST)
        mask_binarized = (mask_original > 0.5).astype(np.uint8)

        # Extraer puntos clave
        point_coords, point_labels = extract_keypoints(mask_binarized, num_points=10)

        mask_tensor_binarized = torch.tensor(mask_binarized, dtype=torch.float32).unsqueeze(0)
        mask_tensor_original = torch.tensor(mask_original, dtype=torch.float32).unsqueeze(0)
        point_tensor_coords = torch.tensor(point_coords, dtype=torch.float32)
        point_tensor_labels = torch.tensor(point_labels, dtype=torch.float32)

        # Liberar memoria intermedia
        del combined_mask, mask_original, mask_binarized
        torch.cuda.empty_cache()

        return img_tensor.squeeze(0), mask_tensor_binarized, mask_tensor_original, point_tensor_coords, point_tensor_labels

    def __len__(self):
        return len(self.patients)

    def get_patient(self, idx):
        return self.patients[idx]