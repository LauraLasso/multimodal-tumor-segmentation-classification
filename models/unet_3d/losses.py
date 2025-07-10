# -*- coding: utf-8 -*-
"""
Funciones de pérdida para segmentación 3D
"""
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """Dice Loss para segmentación"""
    
    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        
        assert probability.shape == targets.shape
        
        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        
        return 1.0 - dice_score

class BCEDiceLoss(nn.Module):
    """Combinación de BCE Loss y Dice Loss"""
    
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.shape == targets.shape
        
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        
        return bce_loss + dice_loss

def get_criterion(loss_type: str = "bce_dice"):
    """Factory para obtener función de pérdida"""
    if loss_type == "bce_dice":
        return BCEDiceLoss()
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Loss type {loss_type} no soportado")
