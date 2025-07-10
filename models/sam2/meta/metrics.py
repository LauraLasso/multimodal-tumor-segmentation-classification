from torchmetrics.classification import JaccardIndex, Dice

iou_metric = JaccardIndex(task="binary")
dice_metric = Dice(num_classes=1, multiclass=False)

def compute_metrics(pred_mask, gt_mask):
    pred_mask_bin = (pred_mask > 0.5).long()
    gt_mask_long = gt_mask.long()
    iou = iou_metric(pred_mask_bin, gt_mask_long).item()
    dice = dice_metric(pred_mask_bin, gt_mask_long).item()
    return iou, dice
