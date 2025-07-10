# -*- coding: utf-8 -*-
"""
Métricas de evaluación para segmentación 3D
"""
import torch
import numpy as np
from typing import Dict, List

def dice_coef_metric(probabilities: torch.Tensor, truth: torch.Tensor, 
                    threshold: float = 0.5, eps: float = 1e-9) -> float:
    """Calcular coeficiente Dice"""
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= threshold).float()
    
    assert predictions.shape == truth.shape
    
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    
    return np.mean(scores)

def jaccard_coef_metric(probabilities: torch.Tensor, truth: torch.Tensor,
                       threshold: float = 0.5, eps: float = 1e-9) -> float:
    """Calcular coeficiente Jaccard (IoU)"""
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= threshold).float()
    
    assert predictions.shape == truth.shape

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    
    return np.mean(scores)

def dice_coef_metric_per_classes(probabilities: np.ndarray, truth: np.ndarray,
                                threshold: float = 0.5, eps: float = 1e-9,
                                classes: List[str] = ['WT', 'TC', 'ET']) -> Dict[str, List[float]]:
    """Calcular Dice por clase"""
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= threshold).astype(np.float32)
    
    assert predictions.shape == truth.shape

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)
                
    return scores

def jaccard_coef_metric_per_classes(probabilities: np.ndarray, truth: np.ndarray,
                                   threshold: float = 0.5, eps: float = 1e-9,
                                   classes: List[str] = ['WT', 'TC', 'ET']) -> Dict[str, List[float]]:
    """Calcular Jaccard por clase"""
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= threshold).astype(np.float32)
    
    assert predictions.shape == truth.shape

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = (prediction * truth_).sum()
            union = (prediction.sum() + truth_.sum()) - intersection + eps
            
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores

class Meter:
    """Medidor de métricas durante entrenamiento"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.dice_scores = []
        self.iou_scores = []
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Actualizar métricas con nuevo batch"""
        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, self.threshold)
        iou = jaccard_coef_metric(probs, targets, self.threshold)
        
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
    
    def get_metrics(self) -> tuple:
        """Obtener métricas promedio"""
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        return dice, iou
    
    def reset(self):
        """Resetear métricas"""
        self.dice_scores = []
        self.iou_scores = []
