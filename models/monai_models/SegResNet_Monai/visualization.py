import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from models.monai_models.SegResNet_Monai.inference import *

def visualizar_pred_vs_gt(paciente_id, input_image, gt_dir, slice_index=60, usar_modelo=True, pred_dir=None, device=None):
    canales = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
    cmap = ["Reds", "Oranges", "Blues"]

    # --- Ground Truth ---
    seg_path = os.path.join(gt_dir, paciente_id, f"{paciente_id}_seg.nii")
    if not os.path.exists(seg_path):
        print(f"No se encontró la máscara real: {seg_path}")
        return

    seg = nib.load(seg_path).get_fdata()
    gt_masks = [
        np.isin(seg, [1, 2, 4]).astype(np.uint8),  # WT
        np.isin(seg, [1, 4]).astype(np.uint8),     # TC
        (seg == 4).astype(np.uint8),               # ET
    ]

    if usar_modelo:
        input_image = input_image.to(device)
        pred = inference(input_image)  # Resultado tensor [1, 3, H, W, D]
        pred = pred[0].detach().cpu().numpy()  # Quitamos batch dim
        pred = (pred > 0.5).astype(np.uint8)
    else:
        # En este caso se esperaría que las predicciones ya estén guardadas por canal
        pred = []
        for i in range(3):
            pred_path = os.path.join(pred_dir, f"{paciente_id}_channel{i}.nii.gz")
            if not os.path.exists(pred_path):
                print(f"No se encontró la máscara predicha: {pred_path}")
                continue
            pred_i = nib.load(pred_path).get_fdata()
            pred.append(pred_i)
        pred = np.stack(pred)

    for i in range(3):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Canal {i}: {canales[i]} - Slice {slice_index}", fontsize=14)

        axs[0].imshow(gt_masks[i][:, :, slice_index], cmap=cmap[i])
        axs[0].set_title("Máscara Ground Truth")
        axs[0].axis("off")

        axs[1].imshow(pred[i][:, :, slice_index], cmap=cmap[i])
        axs[1].set_title("Máscara Predicha")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()