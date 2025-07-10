import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_unetr_prediction(
    model,
    train_loader,
    device,
    checkpoint_path="./UNETR_MONAI_coseno_scheduler.pth",
    patient_idx=2,
    slice_idx=33
):
    """
    Visualiza la predicción de UNETR para un paciente específico,
    mostrando canales de entrada, ground truth, predicción, diferencia y Dice por clase.

    Args:
        model: Modelo UNETR cargado.
        train_loader: DataLoader de entrenamiento (debe tener atributo .dataset).
        device: Dispositivo ('cuda' o 'cpu').
        checkpoint_path: Ruta al checkpoint del modelo.
        patient_idx: Índice del paciente a visualizar.
        slice_idx: Índice del corte axial a mostrar.
    """
    # Cargar pesos y poner en modo evaluación
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    with torch.no_grad():
        val_sample = train_loader.dataset[patient_idx]
        patient_id = val_sample["Id"]
        print(f"Patient ID: {patient_id}")

        # Preparar entrada
        image_np = val_sample["image"].cpu().numpy()
        val_input = torch.from_numpy(image_np).unsqueeze(0).to(device)
        print("Input shape:", val_sample["image"].shape)
        val_output = model(val_input)

        # Ground truth y predicción
        label_vol = torch.tensor(val_sample["label"].cpu().numpy())
        image = val_sample["image"]
        label = val_sample["label"]

        flair = image[3, :, :, slice_idx]
        mask = label[:, :, :, slice_idx]
        pred = torch.sigmoid(val_output[0, :, :, :, slice_idx]).cpu().numpy()
        pred_bin = (pred > 0.5).astype(np.uint8)

        def combine_channels(channels):
            combined = np.zeros_like(channels[0])
            combined[channels[0] > 0] = 1  # WT
            combined[channels[1] > 0] = 2  # TC
            combined[channels[2] > 0] = 3  # ET
            return combined

        mask_combined = combine_channels(mask)
        pred_combined = combine_channels(pred_bin)

        # Diferencias
        diff = np.logical_or.reduce([mask[i] != pred_bin[i] for i in range(3)])
        color_diff = np.zeros((*diff.shape, 4))
        color_diff[diff] = (1, 0, 0, 0.7)

        # Dice por clase
        dice_scores = []
        for i in range(3):
            true_mask = mask[i]
            pred_mask = pred_bin[i]
            intersection = np.sum(true_mask * pred_mask)
            union = np.sum(true_mask) + np.sum(pred_mask)
            dice = 2. * intersection / (union + 1e-8) if union > 0 else 0.0
            dice_scores.append(dice)

        cmap = plt.matplotlib.colors.ListedColormap(['black', 'green', 'red', 'blue'])

        # Visualización
        fig = plt.figure(figsize=(16, 10))
        for i, title in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
            plt.subplot(2, 4, i + 1)
            plt.imshow(image[i, :, :, slice_idx], cmap="gray")
            plt.title(title)
            plt.axis('off')

        plt.subplot(2, 4, 5)
        plt.imshow(flair, cmap="gray", alpha=0.7)
        plt.imshow(mask_combined, cmap=cmap, alpha=0.5)
        plt.title("Ground Truth (All Classes)")
        plt.axis('off')

        plt.subplot(2, 4, 6)
        plt.imshow(flair, cmap="gray", alpha=0.7)
        plt.imshow(pred_combined, cmap=cmap, alpha=0.5)
        plt.title("Prediction (All Classes)")
        plt.axis('off')

        plt.subplot(2, 4, 7)
        plt.imshow(flair, cmap="gray", alpha=0.7)
        plt.imshow(color_diff)
        plt.title("Difference Map")
        plt.axis('off')

        plt.subplot(2, 4, 8)
        plt.bar(["WT", "TC", "ET"], dice_scores, color=['green', 'red', 'blue'])
        plt.ylim(0, 1)
        plt.ylabel("Dice Score")
        plt.title("Dice Scores by Class")

        plt.tight_layout()
        plt.show()
