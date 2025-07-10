# -*- coding: utf-8 -*-
"""
Visualización 2D y 3D para segmentación
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import nibabel as nib
import os
from typing import Optional
import plotly.graph_objects as go
from skimage import measure

def plot_metrics_summary(metrics_df, title: str = "Dice and Jaccard Coefficients", 
                        save_path: Optional[str] = None):
    """Graficar resumen de métricas"""
    colors = ['#35FCFF', '#FF355A', '#96C503', '#C5035B', '#28B463', '#35FFAF']
    palette = sns.color_palette(colors, len(metrics_df.columns))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=metrics_df.mean().index, y=metrics_df.mean(), palette=palette, ax=ax)
    ax.set_xticklabels(metrics_df.columns, fontsize=14, rotation=15)
    ax.set_title(title, fontsize=20)

    for idx, p in enumerate(ax.patches):
        percentage = '{:.1f}%'.format(100 * metrics_df.mean().values[idx])
        x = p.get_x() + p.get_width() / 2 - 0.15
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")

    if save_path:
        fig.savefig(save_path, format="png", pad_inches=0.2, 
                   transparent=False, bbox_inches='tight')
    
    plt.show()

def visualize_segmentation_2d(model: torch.nn.Module, dataloader, num_samples: int = 3, 
                             slice_idx: int = 48, device: str = 'cuda', 
                             patient_id: Optional[str] = None):
    """Visualizar resultados de segmentación en 2D"""
    model.eval()
    class_names = ['Background', 'Whole Tumor', 'Tumor Core', 'Enhancing Tumor']
    colors = ['black', 'green', 'red', 'blue']
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    shown = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device).float()
            masks = batch["mask"].to(device)
            ids = batch["Id"]

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for i in range(images.shape[0]):
                pid = ids[i]

                if patient_id is not None and pid != patient_id:
                    continue
                if patient_id is None and shown >= num_samples:
                    return
                
                print(f"Visualizando paciente: {pid}")

                img = images[i].cpu().numpy()   # (4, D, H, W)
                mask = masks[i].cpu().numpy()   # (3, D, H, W)
                pred = preds[i].cpu().numpy()   # (3, D, H, W)

                # Extraer slice axial
                def get_axial(modality):
                    return modality[:, :, slice_idx]

                t1 = get_axial(img[0])
                t1ce = get_axial(img[1])
                t2 = get_axial(img[2])
                flair = get_axial(img[3])

                # Crear máscaras combinadas
                y_true = np.zeros_like(mask[0, :, :, slice_idx], dtype=np.uint8)
                y_pred = np.zeros_like(pred[0, :, :, slice_idx], dtype=np.uint8)

                y_true[mask[0, :, :, slice_idx] > 0] = 1  # WT
                y_true[mask[1, :, :, slice_idx] > 0] = 2  # TC
                y_true[mask[2, :, :, slice_idx] > 0] = 3  # ET

                y_pred[pred[0, :, :, slice_idx] > 0] = 1
                y_pred[pred[1, :, :, slice_idx] > 0] = 2
                y_pred[pred[2, :, :, slice_idx] > 0] = 3

                # Mapa de diferencias
                diff = (y_true != y_pred)
                color_mask = np.zeros((*diff.shape, 4))
                color_mask[diff] = (1, 0, 0, 0.7)

                # Crear figura
                fig = plt.figure(figsize=(20, 15))
                
                # Modalidades
                for j, modality in enumerate([t1, t1ce, t2, flair]):
                    ax = fig.add_subplot(2, 4, j + 1)
                    ax.imshow(modality, cmap='gray')
                    ax.set_title(['T1', 'T1ce', 'T2', 'FLAIR'][j])
                    ax.axis('off')

                # Ground truth
                ax = fig.add_subplot(2, 4, 5)
                ax.imshow(flair, cmap='gray', alpha=0.7)
                ax.imshow(y_true, cmap=cmap, alpha=0.5)
                ax.set_title('Ground Truth')
                ax.axis('off')

                # Predicción
                ax = fig.add_subplot(2, 4, 6)
                ax.imshow(flair, cmap='gray', alpha=0.7)
                ax.imshow(y_pred, cmap=cmap, alpha=0.5)
                ax.set_title('Prediction')
                ax.axis('off')

                # Mapa de diferencias
                ax = fig.add_subplot(2, 4, 7)
                ax.imshow(flair, cmap='gray', alpha=0.7)
                ax.imshow(color_mask)
                ax.set_title('Difference Map')
                ax.axis('off')

                # Dice scores
                dice_scores = []
                for c in range(1, 4):
                    true_mask = (y_true == c)
                    pred_mask = (y_pred == c)
                    intersection = np.sum(true_mask & pred_mask)
                    union = np.sum(true_mask) + np.sum(pred_mask)
                    dice = 2.0 * intersection / (union + 1e-8) if union > 0 else 1.0
                    dice_scores.append(dice)

                ax = fig.add_subplot(2, 4, 8)
                ax.bar(range(3), dice_scores, color=['green', 'red', 'blue'])
                ax.set_xticks(range(3))
                ax.set_xticklabels(['WT', 'TC', 'ET'])
                ax.set_ylabel('Dice Score')
                ax.set_title('Dice Scores by Class')

                plt.suptitle(f"Patient: {pid}, Slice: {slice_idx}", fontsize=16)
                plt.subplots_adjust(wspace=0.2, hspace=0.2)
                plt.savefig(f'segmentation_vis_{pid}_slice_{slice_idx}.png', 
                           dpi=150, bbox_inches='tight')
                plt.show()

                shown += 1
                return

def save_segmentation_nifti(volume: np.ndarray, pid: str, filename_prefix: str, 
                           output_dir: str = 'volumes'):
    """Guardar volumen de segmentación como NIfTI"""
    os.makedirs(output_dir, exist_ok=True)
    nifti_img = nib.Nifti1Image(volume.astype(np.uint8), affine=np.eye(4))
    save_path = os.path.join(output_dir, f"{filename_prefix}_{pid}.nii.gz")
    nib.save(nifti_img, save_path)
    print(f"Volumen guardado: {save_path}")

def plot_3d_segmentation_interactive(gt_volume: np.ndarray, pred_volume: np.ndarray, 
                                    title: str = "3D Segmentation"):
    """Visualización 3D interactiva con Plotly"""
    color_map = {1: 'green', 2: 'red', 3: 'blue'}
    name_map = {1: 'WT', 2: 'TC', 3: 'ET'}
    
    # Calcular Dice por clase
    def dice_score(gt, pred):
        intersection = np.sum((gt == 1) & (pred == 1))
        return (2.0 * intersection) / (np.sum(gt == 1) + np.sum(pred == 1) + 1e-8)

    print(f"\nDICE COEFFICIENTS for {title}:")
    dice_scores = []
    for label in color_map:
        dice = dice_score(gt_volume == label, pred_volume == label)
        dice_scores.append(dice)
        print(f"  → {name_map[label]}: {dice:.4f}")

    # Gráfico de barras con Dice scores
    labels = ['WT', 'TC', 'ET']
    plt.figure(figsize=(8, 6))
    plt.bar(labels, dice_scores, color=['green', 'red', 'blue'])
    plt.ylabel("Dice Score")
    plt.title("Dice Score by Class")
    plt.ylim(0, 1)
    for i, score in enumerate(dice_scores):
        plt.text(i, score + 0.02, f'{score:.3f}', ha='center')
    plt.show()

    # Visualización 3D
    meshes = []
    for volume in [gt_volume, pred_volume]:
        for label, color in color_map.items():
            binary = (volume == label).astype(np.uint8)
            if np.sum(binary) == 0:
                continue

            try:
                verts, faces, _, _ = measure.marching_cubes(binary, level=0.5)
                
                mesh = go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color=color,
                    opacity=0.5,
                    name=f'{name_map[label]}',
                    visible=True
                )
                meshes.append(mesh)
            except:
                print(f"No se pudo generar mesh para {name_map[label]}")

    if meshes:
        fig = go.Figure(data=meshes)
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data'
            ),
            width=900,
            height=700,
            showlegend=True
        )
        fig.show()
    else:
        print("No se pudieron generar meshes 3D")

def visualize_segmentation_results_3d(model, dataloader, num_samples=3, slice_idx=48, device='cuda', patient_id=None):
    model.eval()
    class_names = ['Background', 'Whole Tumor', 'Tumor Core', 'Enhancing Tumor']
    colors = ['black', 'green', 'red', 'blue']
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    shown = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device).float()
            masks = batch["mask"].to(device)
            ids = batch["Id"]

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for i in range(images.shape[0]):
                pid = ids[i]
                if patient_id is not None and pid != patient_id:
                    continue
                if patient_id is None and shown >= num_samples:
                    return
                print(pid)

                img = images[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                pred = preds[i].cpu().numpy()

                # ========== Generación 2D =============
                def get_axial(modality):
                    return modality[:, :, slice_idx]

                t1, t1ce, t2, flair = (get_axial(img[k]) for k in range(4))

                y_true = np.zeros_like(mask[0, :, :, slice_idx], dtype=np.uint8)
                y_pred = np.zeros_like(pred[0, :, :, slice_idx], dtype=np.uint8)
                y_true[mask[0, :, :, slice_idx] > 0] = 1
                y_true[mask[1, :, :, slice_idx] > 0] = 2
                y_true[mask[2, :, :, slice_idx] > 0] = 3
                y_pred[pred[0, :, :, slice_idx] > 0] = 1
                y_pred[pred[1, :, :, slice_idx] > 0] = 2
                y_pred[pred[2, :, :, slice_idx] > 0] = 3
                diff = (y_true != y_pred)
                color_mask = np.zeros((*diff.shape, 4))
                color_mask[diff] = (1, 0, 0, 0.7)

                fig = plt.figure(figsize=(20, 15))
                for j, modality in enumerate([t1, t1ce, t2, flair]):
                    ax = fig.add_subplot(2, 4, j + 1)
                    ax.imshow(modality, cmap='gray')
                    ax.set_title(['T1', 'T1ce', 'T2', 'FLAIR'][j])
                    ax.axis('off')

                ax = fig.add_subplot(2, 4, 5)
                ax.imshow(flair, cmap='gray', alpha=0.7)
                ax.imshow(y_true, cmap=cmap, alpha=0.5)
                ax.set_title('Ground Truth (All Classes)')
                ax.axis('off')

                ax = fig.add_subplot(2, 4, 6)
                ax.imshow(flair, cmap='gray', alpha=0.7)
                ax.imshow(y_pred, cmap=cmap, alpha=0.5)
                ax.set_title('Prediction (All Classes)')
                ax.axis('off')

                ax = fig.add_subplot(2, 4, 7)
                ax.imshow(flair, cmap='gray', alpha=0.7)
                ax.imshow(color_mask)
                ax.set_title('Difference Map')
                ax.axis('off')

                dice_scores = []
                for c in range(1, 4):
                    true_mask = (y_true == c)
                    pred_mask = (y_pred == c)
                    intersection = np.sum(true_mask & pred_mask)
                    union = np.sum(true_mask) + np.sum(pred_mask)
                    dice = 2.0 * intersection / (union + 1e-8) if union > 0 else 1.0
                    dice_scores.append(dice)

                ax = fig.add_subplot(2, 4, 8)
                ax.bar(range(3), dice_scores, color=['green', 'red', 'blue'])
                ax.set_xticks(range(3))
                ax.set_xticklabels(['WT', 'TC', 'ET'])
                ax.set_ylabel('Dice Score')
                ax.set_title('Dice Scores by Class')

                plt.suptitle(f"Patient: {pid}, Slice: {slice_idx}", fontsize=16)
                plt.subplots_adjust(wspace=0.2, hspace=0.2)
                plt.savefig(f'segmentation_vis_{pid}_slice_{slice_idx}.png', dpi=150, bbox_inches='tight')
                plt.show()

                # ========== Visualización 3D =============
                full_y_true = np.zeros_like(mask[0], dtype=np.uint8)
                full_y_pred = np.zeros_like(pred[0], dtype=np.uint8)

                full_y_true[mask[0] > 0] = 1
                full_y_true[mask[1] > 0] = 2
                full_y_true[mask[2] > 0] = 3

                full_y_pred[pred[0] > 0] = 1
                full_y_pred[pred[1] > 0] = 2
                full_y_pred[pred[2] > 0] = 3

                # Guardar volúmenes como .nii.gz
                # save_segmentation_nifti(full_y_true, pid, 'gt')
                # save_segmentation_nifti(full_y_pred, pid, 'pred')

                # Mostrar visualización interactiva
                # plot_3d_segmentation_interactive(full_y_true, f'Ground Truth 3D - {pid}')
                # plot_3d_segmentation_interactive(full_y_pred, f'Prediction 3D - {pid}')
                plot_3d_segmentation_interactive(full_y_true, full_y_pred, title="Volumetric Segmentation")

                shown += 1
                return

    if patient_id is not None and shown == 0:
        print(f"Paciente con ID '{patient_id}' no encontrado.")
