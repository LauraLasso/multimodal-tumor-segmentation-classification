import torch
import numpy as np
from torchmetrics.classification import JaccardIndex
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_

def train(
    predictor,
    train_dataset,
    val_dataset,
    dice_coeff,
    DiceLoss,
    read_batch,
    n_steps=300,
    accumulation_steps=4,
    lr=1e-4,
    weight_decay=1e-4,
    scheduler_step=100,
    scheduler_gamma=0.2,
    model_save_path="fine_tuned_mask_decoder_memory_encoder_sam2.torch"
):
    """
    Entrenamiento y validación para MedSAM2.
    Args:
        predictor: objeto predictor SAM2 ya inicializado.
        train_dataset: dataset de entrenamiento.
        val_dataset: dataset de validación.
        dice_coeff: función para calcular el Dice.
        DiceLoss: clase de pérdida Dice.
        read_batch: función para obtener un batch del dataset.
        n_steps: número de pasos de entrenamiento.
        accumulation_steps: pasos para acumulación de gradientes.
        lr: learning rate.
        weight_decay: weight decay para el optimizador.
        scheduler_step: pasos para el scheduler.
        scheduler_gamma: factor de reducción del scheduler.
        model_save_path: ruta para guardar el modelo.
    Returns:
        dict con historial de métricas.
    """
    # Métricas e historial
    losses, val_losses = [], []
    iou_scores, val_iou_scores = [], []
    dice_scores, val_dice_scores = [], []
    true_labels, val_true_labels = [], []
    pred_scores, val_pred_scores = [], []

    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    mean_iou = 0
    mean_dice = 0
    mean_val_iou = 0
    mean_val_dice = 0

    iou_metric = JaccardIndex(task="binary").cuda()
    criterion_dice = DiceLoss()

    for step in range(1, n_steps + 1):
        with amp.autocast():
            image, mask, point_coords, num_masks = read_batch(train_dataset, visualize_data=False)
            if image is None or mask is None or num_masks == 0:
                continue

            point_labels = np.ones((num_masks, 1), dtype=np.float32)

            if not isinstance(point_coords, np.ndarray) or not isinstance(point_labels, np.ndarray):
                continue
            if point_coords.size == 0 or point_labels.size == 0:
                continue

            point_coords = point_coords.reshape(1, -1, 2)
            point_labels = point_labels.reshape(1, -1)

            point_coords = torch.tensor(point_coords, dtype=torch.float32).cuda()
            point_labels = torch.tensor(point_labels, dtype=torch.float32).cuda()

            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                point_coords, point_labels, box=None, mask_logits=None, normalize_coords=True
            )

            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                continue

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None,
            )

            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )

            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])

            iou_value = iou_metric(prd_mask, gt_mask).item()
            dice_value = dice_coeff(prd_mask, gt_mask).item()
            seg_loss = criterion_dice(prd_masks[:, 0], gt_mask)

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = (seg_loss + score_loss * 0.05) / accumulation_steps

            losses.append(loss.item())
            iou_scores.append(iou_value)
            dice_scores.append(dice_value)
            true_labels.extend(gt_mask.cpu().numpy().flatten().tolist())
            pred_scores.extend(prd_mask.detach().cpu().numpy().flatten().tolist())

            scaler.scale(loss).backward()
            clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                predictor.model.zero_grad()

            scheduler.step()

            mean_iou = mean_iou * 0.99 + 0.01 * iou_value
            mean_dice = mean_dice * 0.99 + 0.01 * dice_value

            print(f"Training - IoU: {mean_iou:.4f} | Dice: {mean_dice:.4f} | Loss: {losses[-1]:.4f}")

            # Validación en cada paso
            val_loss = 0
            with torch.no_grad():
                for i in range(len(val_dataset)):
                    val_image, val_mask, val_point_coords, val_num_masks = read_batch(val_dataset, visualize_data=False)
                    if val_image is None or val_mask is None or val_num_masks == 0:
                        continue

                    val_point_labels = np.ones((val_num_masks, 1), dtype=np.float32)
                    val_point_coords = val_point_coords.reshape(1, -1, 2)
                    val_point_labels = val_point_labels.reshape(1, -1)

                    val_point_coords = torch.tensor(val_point_coords, dtype=torch.float32).cuda()
                    val_point_labels = torch.tensor(val_point_labels, dtype=torch.float32).cuda()

                    predictor.set_image(val_image)
                    val_mask_input, val_unnorm_coords, val_labels, val_unnorm_box = predictor._prep_prompts(
                        val_point_coords, val_point_labels, box=None, mask_logits=None, normalize_coords=True
                    )

                    if val_unnorm_coords is None or val_labels is None or val_unnorm_coords.shape[0] == 0 or val_labels.shape[0] == 0:
                        continue

                    val_sparse_embeddings, val_dense_embeddings = predictor.model.sam_prompt_encoder(
                        points=(val_unnorm_coords, val_labels), boxes=None, masks=None,
                    )

                    val_high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

                    val_low_res_masks, val_prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=val_sparse_embeddings,
                        dense_prompt_embeddings=val_dense_embeddings,
                        multimask_output=True,
                        repeat_image=False,
                        high_res_features=val_high_res_features
                    )

                    val_prd_masks = predictor._transforms.postprocess_masks(val_low_res_masks, predictor._orig_hw[-1])
                    val_gt_mask = torch.tensor(val_mask.astype(np.float32)).cuda()
                    val_prd_mask = torch.sigmoid(val_prd_masks[:, 0])

                    val_iou_value = iou_metric(val_prd_mask, val_gt_mask).item()
                    val_dice_value = dice_coeff(val_prd_mask, val_gt_mask).item()
                    val_seg_loss = criterion_dice(val_prd_masks[:, 0], val_gt_mask)

                    val_inter = (val_gt_mask * (val_prd_mask > 0.5)).sum(1).sum(1)
                    val_iou = val_inter / (val_gt_mask.sum(1).sum(1) + (val_prd_mask > 0.5).sum(1).sum(1) - val_inter)
                    val_score_loss = torch.abs(val_prd_scores[:, 0] - val_iou).mean()

                    val_total_loss = val_seg_loss + val_score_loss * 0.05
                    val_loss += val_total_loss.item()

                    val_iou_scores.append(val_iou_value)
                    val_dice_scores.append(val_dice_value)
                    val_true_labels.extend(val_gt_mask.cpu().numpy().flatten().tolist())
                    val_pred_scores.extend(val_prd_mask.detach().cpu().numpy().flatten().tolist())

                val_losses.append(val_loss / len(val_dataset))

            mean_val_iou = np.mean(val_iou_scores)
            mean_val_dice = np.mean(val_dice_scores)
            print(f"Validation - IoU: {mean_val_iou:.4f} | Dice: {mean_val_dice:.4f} | Loss: {val_losses[-1]:.4f}")

    # Guardar el modelo entrenado
    torch.save(predictor.model.state_dict(), model_save_path)
    print(f"Modelo guardado en {model_save_path}")

    # Devuelve el historial de métricas
    return {
        "losses": losses,
        "val_losses": val_losses,
        "iou_scores": iou_scores,
        "val_iou_scores": val_iou_scores,
        "dice_scores": dice_scores,
        "val_dice_scores": val_dice_scores,
        "true_labels": true_labels,
        "val_true_labels": val_true_labels,
        "pred_scores": pred_scores,
        "val_pred_scores": val_pred_scores
    }
