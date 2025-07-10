import cv2
import numpy as np
import torch
import nibabel as nib
import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def read_batch(dataset, visualize_data=False):

    # Seleccionar una entrada aleatoria del dataset
    idx = np.random.randint(len(dataset))
    sample = dataset[idx]

    if sample is None:
        print(f"Error: No se pudo leer la muestra en índice {idx}.")
        return None, None, None, 0

    # Extraer imagen, máscara y puntos clave
    image, mask_binarized, _, point_coords, point_labels = sample

    # Asegurar que la imagen sea 2D o 3D en formato correcto
    image = image.numpy()
    if image.shape[0] == 3:  # Convertir de (C, H, W) → (H, W, C) si es RGB
        image = np.transpose(image, (1, 2, 0))

    mask = mask_binarized.numpy().astype(np.uint8)  # Convertir la máscara a uint8

    # Si no hay puntos clave, devolver datos vacíos
    if point_coords.shape[0] == 0:
        print(f"Advertencia: No se encontraron puntos clave en la muestra {idx}, saltando...")
        return image, mask, np.empty((0, 2)), 0

    # Convertir puntos clave a numpy
    point_coords = point_coords.numpy().astype(np.int32)  # Asegurar enteros
    num_masks = len(point_coords)  # Número de puntos clave detectados

    return image, mask, point_coords, num_masks

def load_nifti(filepath):
    nii_img = nib.load(filepath)
    img_data = nii_img.get_fdata(dtype=np.float32)  # Evita conversión a doble precisión
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data) + 1e-6)  # Normalización segura
    return img_data

def combine_masks(mask_paths):
    combined_mask = None
    for mask_path in mask_paths:
        mask = load_nifti(mask_path).astype(np.float32)
        if combined_mask is None:
            combined_mask = mask
        else:
            np.maximum(combined_mask, mask, out=combined_mask)  # Evita copias innecesarias
        del mask  # Liberar memoria de la máscara temporal
    return combined_mask

def load_nifti_slice(filepath, axis=2):
    nii_img = nib.load(filepath)
    data = nii_img.get_fdata(dtype=np.float32)
    slice_index = data.shape[axis] // 2
    if axis == 0:
        slice_2d = data[slice_index, :, :]
    elif axis == 1:
        slice_2d = data[:, slice_index, :]
    else:
        slice_2d = data[:, :, slice_index]
    # Normaliza y devuelve
    slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-6)
    print("Slice shape:", slice_2d.shape)
    print("Slice size (MB):", slice_2d.nbytes / 1e6)
    return slice_2d

def extract_keypoints(mask, num_points=10):
    """Extrae puntos clave dentro de la máscara binarizada con erosión para evitar ruido en los bordes."""
    eroded_mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)  # Aplicar erosión
    coords = np.argwhere(eroded_mask > 0)  # Encuentra píxeles donde la máscara es 1

    if len(coords) > 0:
        print(':)')
        num_samples = min(num_points, len(coords))  # Limita la cantidad de puntos
        selected_points = np.random.choice(len(coords), num_samples, replace=False)
        point_coords = coords[selected_points]  # Seleccionar puntos clave dentro de la máscara
        point_labels = np.ones((num_samples, 1))  # Todos los puntos pertenecen a la clase 1
    else:
        point_coords = np.empty((0, 2))  # Evita errores si no hay puntos
        point_labels = np.empty((0, 1))
        print('No se encontraron puntos claves, saltando...')

    return point_coords, point_labels

def combine_masks_visualization(mask_list):
    mask_list = [(mask > 0.5).astype(np.uint8) for mask in mask_list]

    for i, mask in enumerate(mask_list):
        print(f"--> Máscara {i+1} - Shape: {mask.shape}, Valores únicos: {np.unique(mask)}")

    combined_mask = np.zeros_like(mask_list[0], dtype=np.uint8)
    for mask in mask_list:
        np.maximum(combined_mask, mask, out=combined_mask)  # Fusión segura

    print(f"--> Máscara combinada - Shape: {combined_mask.shape}, Valores únicos: {np.unique(combined_mask)}")

    return combined_mask


def visualize_masks(dataset, idx):
    patient_id = dataset.get_patient(idx)
    patient_path = os.path.join(dataset.dataset_path, patient_id)
    files = sorted(os.listdir(patient_path))
    mask_files = [os.path.join(patient_path, f) for f in files if "mask" in f]

    masks = [load_nifti(mask_path) for mask_path in mask_files]

    print(f"Visualizando paciente: {patient_id} - {len(masks)} máscaras detectadas.")

    fig, axes = plt.subplots(1, len(masks) + 1, figsize=(25, 10))

    for i, mask in enumerate(masks):
        mask_slice = dataset.extract_slice(mask)
        axes[i].imshow(mask_slice, cmap="gray")
        axes[i].set_title(f"Máscara {i+1}", fontsize=18)
        axes[i].axis("off")

    combined_mask = combine_masks_visualization(masks)
    mask_original = dataset.extract_slice(combined_mask)

    axes[-1].imshow(mask_original, cmap="jet")
    axes[-1].set_title("Máscara Combinada", fontsize=18)
    axes[-1].axis("off")

    plt.subplots_adjust(wspace=0.3)
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def visualize_sample(image, mask_original, mask_binarized, point_coords):
    plt.figure(figsize=(20, 5))

    # Convertir tensores a NumPy si es necesario
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(mask_original):
        mask_original = mask_original.cpu().numpy()
    if torch.is_tensor(mask_binarized):
        mask_binarized = mask_binarized.cpu().numpy()
    if torch.is_tensor(point_coords):
        point_coords = point_coords.cpu().numpy()

    # Asegurar que todas las imágenes sean 2D (H, W) o (H, W, C)
    image = image.squeeze()  # (1, H, W) → (H, W) o (3, H, W) → (3, H, W)
    mask_original = mask_original.squeeze()
    mask_binarized = mask_binarized.squeeze()

    # Si la imagen es RGB (3 canales), reorganizar de (C, H, W) → (H, W, C)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # (3, H, W) → (H, W, 3)

    # Mostrar la imagen original
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap="gray" if image.ndim == 2 else None)
    plt.title("Imagen Original")
    plt.axis("off")

    # Mostrar la máscara original antes de binarización
    plt.subplot(1, 4, 2)
    plt.imshow(mask_original, cmap="gray")
    plt.title("Máscara Original")
    plt.axis("off")

    # Mostrar la máscara binarizada final
    plt.subplot(1, 4, 3)
    plt.imshow(mask_binarized, cmap="gray")
    plt.title("Máscara Binarizada")
    plt.axis("off")

    # Mostrar la máscara con puntos clave
    plt.subplot(1, 4, 4)
    plt.imshow(mask_binarized, cmap="gray")
    plt.title("Máscara con Puntos Clave")

    # Dibujar puntos clave en la máscara binarizada solo si existen
    if point_coords.shape[0] > 0:
        point_coords = point_coords.astype(int)  # Convertir a enteros para matplotlib
        colors = list(mcolors.TABLEAU_COLORS.values())  # Colores para los puntos
        for i, point in enumerate(point_coords):
            plt.scatter(point[1], point[0], c=colors[i % len(colors)], s=80, label=f'Punto {i+1}')
    else:
        print("No se detectaron puntos clave en esta segmentación.")

    plt.axis("off")
    plt.show()

def extract_patient_id(folder_name):
    match = re.match(r"BCBM-RadioGenomics-(\d+)-\d+", folder_name)
    return int(match.group(1)) if match else None

def select_patient(patient_id, patients=None, train_dataset=None):
    # Buscar la carpeta del paciente que coincida con el ID
    selected_patient = next((p for p in patients if extract_patient_id(p) == patient_id), None)
    print(selected_patient)

    if selected_patient is None:
        print(f"Error: Paciente con ID '{patient_id}' no encontrado en la base de datos.")
        return

    idx = patients.index(selected_patient)  # Obtener el índice del paciente en la lista

    # Cargar datos del paciente específico
    sample = train_dataset[idx]
    if sample is None:
        print(f"Error al cargar los datos del paciente '{selected_patient}'")
        return

    # Extraer los datos
    image, mask_binarized, mask_original, point_coords, _ = sample

    # Visualizar la imagen del paciente seleccionado
    visualize_sample(image, mask_original, mask_binarized, point_coords)