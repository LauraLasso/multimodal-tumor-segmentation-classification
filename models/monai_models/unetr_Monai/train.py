import os
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def train_unetr_model(
    model,
    train_loader,
    val_loader,
    device,
    root_dir="./",
    max_epochs=50,
    val_interval=1,
    log_filename="metrics_log.txt"
):
    """
    Entrenamiento y validación para UNETR MONAI con logging de métricas y guardado de mejor modelo.
    """
    # Configuración de pérdida, optimizador y scheduler
    loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    log_path = os.path.join(root_dir, log_filename)
    with open(log_path, "w") as f:
        f.write("Epoch\tTrainLoss\tValLoss\tTrainDice\tTrain_TC\tTrain_WT\tTrain_ET\t"
                "ValDice\tVal_TC\tVal_WT\tVal_ET\tLR\n")

    torch.cuda.empty_cache()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        train_dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
        train_tc = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
        train_wt = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
        train_et = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.3)])

        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                preds = post_trans(outputs)
                train_dice_metric(y_pred=preds, y=labels)
                train_tc(y_pred=preds[:, 0:1], y=labels[:, 0:1])
                train_wt(y_pred=preds[:, 1:2], y=labels[:, 1:2])
                train_et(y_pred=preds[:, 2:3], y=labels[:, 2:3])

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        train_dice, _ = train_dice_metric.aggregate()
        train_tc_val, _ = train_tc.aggregate()
        train_wt_val, _ = train_wt.aggregate()
        train_et_val, _ = train_et.aggregate()
        train_dice_metric.reset()
        train_tc.reset()
        train_wt.reset()
        train_et.reset()

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} train dice: {train_dice.item():.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            val_steps = 0
            with torch.no_grad():
                dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
                dice_metric_tc = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
                dice_metric_wt = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
                dice_metric_et = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)

                for val_data in val_loader:
                    val_steps += 1
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = model(val_inputs)
                    val_loss += loss_function(val_outputs, val_labels).item()

                    val_outputs = post_trans(val_outputs)
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_tc(y_pred=val_outputs[:, 0:1], y=val_labels[:, 0:1])
                    dice_metric_wt(y_pred=val_outputs[:, 1:2], y=val_labels[:, 1:2])
                    dice_metric_et(y_pred=val_outputs[:, 2:3], y=val_labels[:, 2:3])

                val_loss /= val_steps
                value, _ = dice_metric.aggregate()
                metric = value.mean().item()
                dice_metric.reset()

                value_tc, _ = dice_metric_tc.aggregate()
                metric_tc = value_tc.item()
                dice_metric_tc.reset()

                value_wt, _ = dice_metric_wt.aggregate()
                metric_wt = value_wt.item()
                dice_metric_wt.reset()

                value_et, _ = dice_metric_et.aggregate()
                metric_et = value_et.item()
                dice_metric_et.reset()

                metric_values.append(metric)
                metric_values_tc.append(metric_tc)
                metric_values_wt.append(metric_wt)
                metric_values_et.append(metric_et)

                scheduler.step()

                current_lr = optimizer.param_groups[0]['lr']

                with open(log_path, "a") as f:
                    f.write(f"{epoch+1}\t{epoch_loss:.4f}\t{val_loss:.4f}\t{train_dice.item():.4f}\t"
                            f"{train_tc_val.item():.4f}\t{train_wt_val.item():.4f}\t{train_et_val.item():.4f}\t"
                            f"{metric:.4f}\t{metric_tc:.4f}\t{metric_wt:.4f}\t{metric_et:.4f}\t{current_lr:.6f}\n")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                    print("saved new best metric model")

                print(
                    f"epoch {epoch + 1} - train dice: {train_dice.item():.4f} | val loss: {val_loss:.4f} | val dice: {metric:.4f} "
                    f"tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f} | LR: {current_lr:.6f} \n"
                    f"best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )

    torch.save(model.state_dict(), os.path.join(root_dir, "UNETR_coseno_scheduler.pth"))
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    return {
        "epoch_loss_values": epoch_loss_values,
        "metric_values": metric_values,
        "metric_values_tc": metric_values_tc,
        "metric_values_wt": metric_values_wt,
        "metric_values_et": metric_values_et,
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch
    }
