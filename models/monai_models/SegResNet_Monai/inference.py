import os
import torch
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import decollate_batch
from tqdm import tqdm

def inference(model, val_loader, output_dir, device):
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Inferencia")):
            inputs = batch["image"].to(device).float()
            ids = batch["Id"]
            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5
            )
            outputs = [post_trans(i) for i in decollate_batch(outputs)]
            for j, mask in enumerate(outputs):
                pred_mask = mask.cpu().numpy()
                patient_id = ids[j]
                for c in range(pred_mask.shape[0]):
                    out_path = os.path.join(output_dir, f"{patient_id}_channel{c}.nii.gz")
                    nib.save(nib.Nifti1Image(pred_mask[c].astype(np.uint8), affine=np.eye(4)), out_path)
                print(f"[✓] Guardada segmentación para paciente {patient_id}")
