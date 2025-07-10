import os
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from skimage import measure
import plotly.io as pio
import matplotlib.pyplot as plt

pio.renderers.default = "browser"

def save_segmentation_nifti(volume, pid, filename_prefix, output_dir='volumes'):
    """Save segmentation volume as NIfTI file"""
    os.makedirs(output_dir, exist_ok=True)
    nifti_img = nib.Nifti1Image(volume.astype(np.uint8), affine=np.eye(4))
    save_path = os.path.join(output_dir, f"{filename_prefix}_{pid}.nii.gz")
    nib.save(nifti_img, save_path)
    print(f"[INFO] Saved {filename_prefix} volume for patient {pid} → {save_path}")

def plot_3d_segmentation_interactive(gt_volume, pred_volume, title):
    """Plot interactive 3D segmentation with Plotly"""
    color_map = {2: 'green', 1: 'red', 3: 'blue'}
    name_map = {2: 'WT', 1: 'TC', 3: 'ET'}
    
    def dice_score(gt, pred):
        intersection = np.sum((gt == 1) & (pred == 1))
        return (2.0 * intersection) / (np.sum(gt == 1) + np.sum(pred == 1) + 1e-8)
    
    print(f"\n DICE COEFFICIENTS for {title}:")
    dice_scores = []
    for label in color_map:
        dice = dice_score(gt_volume == label, pred_volume == label)
        dice_scores.append(dice)
        print(f"  → {name_map[label]}: {dice:.4f}")
    
    labels = ['WT', 'TC', 'ET']
    plt.figure(figsize=(4, 8))
    plt.bar(labels, dice_scores, color=['green', 'red', 'blue'])
    plt.ylabel("Dice Score")
    plt.title("Dice by class")
    plt.show()
    
    meshes = []
    labels = []
    volumes = [gt_volume, pred_volume]
    
    for volume in volumes:
        labels = []
        for label, color in color_map.items():
            binary = (volume == label).astype(np.uint8)
            if np.sum(binary) == 0:
                continue
            
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
            labels.append(name_map[label])
        
        fig = go.Figure(data=meshes)
        
        # Create visibility menu
        buttons = []
        n = len(labels)
        for i in range(2 ** n):
            visible = []
            label = []
            for j in range(n):
                show = bool(i & (1 << j))
                visible.append(show)
                if show:
                    label.append(labels[j])
            label_str = '+'.join(label) or "None"
            buttons.append(dict(
                label=label_str,
                method="update",
                args=[{"visible": visible},
                    {"title": f"{title} | Showing: {label_str}"}]
            ))
        
        fig.update_layout(
            updatemenus=[dict(
                type="dropdown",
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )],
            title=title,
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data'
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            width=900,
            height=850,
            margin=dict(r=0, l=0, b=0, t=40),
            showlegend=False
        )
        
        fig.show()

def visualize_segmentation_results(model, val_generator, num_samples=3, slice_idx=48, patient_id=None):
    """Visualize segmentation results for specific patients"""
    class_names = ['Background', 'Necrotic', 'Edema', 'Enhancing']
    colors = ['black', 'red', 'green', 'blue']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    shown = 0
    found = False
    
    for batch_idx in range(len(val_generator)):
        x_batch, y_batch = val_generator[batch_idx]
        
        batch_samples = val_generator.data_list[
            batch_idx * val_generator.batch_size: (batch_idx + 1) * val_generator.batch_size
        ]
        
        for i, sample in enumerate(batch_samples):
            pid = sample.get("Brats20ID", sample.get("subject", None))
            
            if patient_id is not None and pid != patient_id:
                continue
            
            found = True
            
            x = x_batch[i]
            y = y_batch[i]
            
            x_input = np.expand_dims(x, axis=0).astype(np.float32)
            y_pred = model.predict(x_input)
            y_pred_classes = np.argmax(y_pred, axis=-1)[0]
            
            flair_slice = x[:, :, slice_idx, 3]
            
            gt_one_hot = y[:, :, slice_idx, :]
            gt_combined = np.argmax(gt_one_hot, axis=-1).astype(np.uint8)
            
            pred_slice = y_pred_classes[:, :, slice_idx]
            
            # Difference map
            mask_valid = gt_combined > 0
            diff = (gt_combined != pred_slice) & mask_valid
            color_mask = np.zeros((*diff.shape, 4))
            color_mask[diff] = (1, 0, 0, 0.7)
            
            # Plot results
            fig = plt.figure(figsize=(20, 15))
            modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
            
            for idx in range(4):
                ax = plt.subplot(2, 4, idx + 1)
                ax.imshow(x[:, :, slice_idx, idx], cmap='gray')
                ax.set_title(modality_names[idx])
                ax.axis('off')
            
            ax = plt.subplot(2, 4, 5)
            ax.imshow(flair_slice, cmap='gray', alpha=0.7)
            ax.imshow(gt_combined, cmap=cmap, alpha=0.5, interpolation='nearest')
            ax.set_title('Ground Truth')
            ax.axis('off')
            
            ax = plt.subplot(2, 4, 6)
            ax.imshow(flair_slice, cmap='gray', alpha=0.7)
            ax.imshow(pred_slice, cmap=cmap, alpha=0.5, interpolation='nearest')
            ax.set_title('Prediction')
            ax.axis('off')
            
            ax = plt.subplot(2, 4, 7)
            ax.imshow(flair_slice, cmap='gray', alpha=0.7)
            ax.imshow(color_mask)
            ax.set_title('Difference')
            ax.axis('off')
            
            # Dice scores per class
            dice_scores = []
            class_labels = ['WT', 'TC', 'ET']
            class_colors = ['green', 'red', 'blue']
            
            for c in [1, 2, 3]:
                true_mask = gt_combined == c
                pred_mask = pred_slice == c
                intersection = np.sum(true_mask & pred_mask)
                union = np.sum(true_mask) + np.sum(pred_mask)
                dice = 2.0 * intersection / (union + 1e-8) if union > 0 else 1.0
                dice_scores.append(dice)
            
            ax = plt.subplot(2, 4, 8)
            ax.bar(class_labels, dice_scores, color=class_colors)
            ax.set_ylabel('Dice Score')
            ax.set_title('Dice Scores')
            
            plt.suptitle(f"Patient ID: {pid}, Slice: {slice_idx}", fontsize=16)
            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            plt.tight_layout()
            plt.show()
            
            # 3D visualization
            gt_vol = np.argmax(y, axis=-1).astype(np.uint8)
            pred_vol = y_pred_classes.astype(np.uint8)
            
            plot_3d_segmentation_interactive(gt_vol, pred_vol, title="Volumetric Segmentation")
            
            shown += 1
            if patient_id is not None:
                return
            if shown >= num_samples:
                return
    
    if patient_id is not None and not found:
        print(f"Paciente con ID '{patient_id}' no encontrado.")
