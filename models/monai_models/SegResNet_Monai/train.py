import os
import time
import torch

def train_segmentation_model(
    model,
    train_loader,
    val_loader,
    train_ds,
    optimizer,
    scaler,
    loss_function,
    lr_scheduler,
    dice_metric,
    dice_metric_batch,
    inference,
    post_trans,
    decollate_batch,
    device,
    root_dir,
    max_epochs=300,
    val_interval=1
):
    """
    Entrena un modelo de segmentación (ej. SegResNet/UNETR MONAI) y realiza validación periódica.
    Guarda el modelo con mejor métrica de validación.

    Args:
        model: Modelo PyTorch.
        train_loader: DataLoader de entrenamiento.
        val_loader: DataLoader de validación.
        train_ds: Dataset de entrenamiento (para tamaño).
        optimizer: Optimizador PyTorch.
        scaler: GradScaler para AMP.
        loss_function: Función de pérdida.
        lr_scheduler: Scheduler de learning rate.
        dice_metric: Métrica Dice global.
        dice_metric_batch: Métrica Dice por clase.
        inference: Función de inferencia (ej. sliding_window_inference).
        post_trans: Transformación post-inferencia.
        decollate_batch: Función para separar batch en listas.
        device: Dispositivo ('cuda' o 'cpu').
        root_dir: Carpeta donde guardar el mejor modelo.
        max_epochs: Número de épocas.
        val_interval: Frecuencia de validación (en épocas).

    Returns:
        dict con historial de métricas y pérdidas.
    """
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    total_start = time.time()
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs = batch_data["image"].to(device).float()
            labels = batch_data["label"].to(device).float()

            optimizer.zero_grad()

            # Si da problemas usar autocast, pon enabled=False o quítalo completamente
            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}, "
                f"step time: {(time.time() - step_start):.4f}s"
            )

            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()

        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device).float()
                    val_labels = val_data["label"].to(device).float()

                    # Asegúrate de que inference() usa un roi_size más pequeño
                    val_outputs = inference(val_inputs)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()

                metric_tc = metric_batch[0].item()
                metric_values_tc.append(metric_tc)
                metric_wt = metric_batch[1].item()
                metric_values_wt.append(metric_wt)
                metric_et = metric_batch[2].item()
                metric_values_et.append(metric_et)

                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    torch.save(
                        model.state_dict(),
                        os.path.join(root_dir, "best_metric_model.pth"),
                    )
                    print("Saved new best metric model")

                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f} | "
                    f"tc: {metric_tc:.4f}, wt: {metric_wt:.4f}, et: {metric_et:.4f}\n"
                    f"best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
                del val_inputs, val_labels, val_outputs
                torch.cuda.empty_cache()

        print(f"Epoch {epoch + 1} time: {(time.time() - epoch_start):.4f}s")

    total_time = time.time() - total_start
    print(f"Total training time: {total_time:.2f}s")

    return {
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "best_metrics_epochs_and_time": best_metrics_epochs_and_time,
        "epoch_loss_values": epoch_loss_values,
        "metric_values": metric_values,
        "metric_values_tc": metric_values_tc,
        "metric_values_wt": metric_values_wt,
        "metric_values_et": metric_values_et,
        "total_time": total_time,
    }
