import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, y_pred, y_true):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()
        intersection = torch.sum(y_pred * y_true)
        dice_score = (2. * intersection + self.smooth) / (torch.sum(y_pred) + torch.sum(y_true) + self.smooth)
        return 1 - dice_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
    def forward(self, y_pred, y_true):
        epsilon = 1e-8
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * torch.log(y_pred)
        loss = self.alpha * torch.pow(1 - y_pred, self.gamma) * cross_entropy
        if self.reduce:
            return loss.mean()
        else:
            return loss.sum()

class BoundaryLoss(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(BoundaryLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, y_pred, y_true):
        y_pred_boundary = self.get_boundary(y_pred)
        y_true_boundary = self.get_boundary(y_true)
        boundary_loss = torch.sum(torch.abs(y_pred_boundary - y_true_boundary))
        return self.alpha * boundary_loss + self.beta * torch.mean(torch.abs(y_pred - y_true))
    def get_boundary(self, mask):
        kernel = torch.tensor([[1, 1, 1], [1, -7, 1], [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        boundary = torch.nn.functional.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)
        return boundary.squeeze(0).squeeze(0)
