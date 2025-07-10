# -*- coding: utf-8 -*-
"""
Funciones de inferencia y evaluación
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, List
from models.unet_3d.metrics import dice_coef_metric_per_classes, jaccard_coef_metric_per_classes
from models.unet_3d.config import get_device

def compute_scores_per_classes(model: torch.nn.Module, dataloader, 
                              classes: List[str] = ['WT', 'TC', 'ET']) -> tuple:
    """Computar scores por clase en un dataloader"""
    device = get_device()
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            imgs, targets = data['image'], data['mask']
            imgs, targets = imgs.to(device).float(), targets.to(device)
            logits = model(imgs)
            logits = logits.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            dice_scores = dice_coef_metric_per_classes(logits, targets)
            iou_scores = jaccard_coef_metric_per_classes(logits, targets)

            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])

            for key in iou_scores.keys():
                iou_scores_per_classes[key].extend(iou_scores[key])

    return dice_scores_per_classes, iou_scores_per_classes

def evaluate_model(model: torch.nn.Module, dataloader, classes: List[str] = ['WT', 'TC', 'ET']) -> pd.DataFrame:
    """Evaluar modelo y retornar DataFrame con métricas"""
    dice_scores, iou_scores = compute_scores_per_classes(model, dataloader, classes)
    
    # Crear DataFrames
    dice_df = pd.DataFrame(dice_scores)
    dice_df.columns = [f'{cls} dice' for cls in classes]
    
    iou_df = pd.DataFrame(iou_scores)
    iou_df.columns = [f'{cls} jaccard' for cls in classes]
    
    # Combinar métricas
    metrics_df = pd.concat([dice_df, iou_df], axis=1, sort=True)
    
    # Reordenar columnas
    ordered_cols = []
    for cls in classes:
        ordered_cols.extend([f'{cls} dice', f'{cls} jaccard'])
    
    metrics_df = metrics_df.loc[:, ordered_cols]
    
    return metrics_df

def predict_single_volume(model: torch.nn.Module, volume: torch.Tensor, device: str = None) -> np.ndarray:
    """Predicción para un solo volumen"""
    if device is None:
        device = get_device()
    
    model.eval()
    with torch.no_grad():
        volume = volume.to(device).float()
        if len(volume.shape) == 4:  # Añadir dimensión de batch si es necesario
            volume = volume.unsqueeze(0)
        
        logits = model(volume)
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).float()
        
        return predictions.cpu().numpy()

def batch_inference(model: torch.nn.Module, dataloader) -> Dict[str, np.ndarray]:
    """Inferencia en batch para todo un dataloader"""
    device = get_device()
    model.eval()
    
    all_predictions = []
    all_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device).float()
            ids = batch["Id"]
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_ids.extend(ids)
    
    return {
        "predictions": np.array(all_predictions),
        "ids": all_ids
    }
