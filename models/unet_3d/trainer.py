# -*- coding: utf-8 -*-
"""
Clase Trainer para entrenamiento de 3D U-Net
"""
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from IPython.display import clear_output
from typing import Dict, List

from models.unet_3d.dataset import get_dataloader
from models.unet_3d.metrics import Meter
from models.unet_3d.config import get_device

class Trainer:
    """Entrenador para 3D U-Net"""
    
    def __init__(self, net: nn.Module, dataset, criterion: nn.Module, lr: float,
                 accumulation_steps: int, batch_size: int, fold: int, num_epochs: int,
                 path_to_csv: str, display_plot: bool = True):
        
        self.device = get_device()
        print(f"Device: {self.device}")
        
        self.display_plot = display_plot
        self.net = net.to(self.device)
        self.criterion = criterion
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=2, verbose=True)
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "val"]
        self.num_epochs = num_epochs

        # Crear DataLoaders
        self.dataloaders = {
            phase: get_dataloader(
                dataset=dataset,
                path_to_csv=path_to_csv,
                phase=phase,
                fold=fold,
                batch_size=batch_size,
                num_workers=0
            )
            for phase in self.phases
        }
        
        # Métricas
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}
         
    def _compute_loss_and_outputs(self, images: torch.Tensor, targets: torch.Tensor):
        """Computar pérdida y outputs"""
        images = images.to(self.device).float()
        targets = targets.to(self.device).float()
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits
    
    def _do_epoch(self, epoch: int, phase: str) -> float:
        """Ejecutar una época de entrenamiento o validación"""
        print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}")

        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        
        for itr, data_batch in enumerate(dataloader):
            images, targets = data_batch['image'], data_batch['mask']
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            running_loss += loss.item()
            meter.update(logits.detach().cpu(), targets.detach().cpu())
            
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou = meter.get_metrics()
        
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)

        return epoch_loss
        
    def run(self):
        """Ejecutar entrenamiento completo"""
        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train")
            
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                self.scheduler.step(val_loss)
            
            if self.display_plot:
                self._plot_train_history()
                
            if val_loss < self.best_loss:
                print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), "best_model.pth")
            print()
            
        self._save_train_history()

    def _plot_train_history(self):
        """Graficar historial de entrenamiento"""
        data = [self.losses, self.dice_scores, self.jaccard_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"train loss {self.losses['train'][-1]:.4f}\nval loss {self.losses['val'][-1]:.4f}",
            f"train dice {self.dice_scores['train'][-1]:.4f}\nval dice {self.dice_scores['val'][-1]:.4f}",
            f"train jaccard {self.jaccard_scores['train'][-1]:.4f}\nval jaccard {self.jaccard_scores['val'][-1]:.4f}",
        ]
        
        clear_output(True)
        with plt.style.context("default"):
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            for i, ax in enumerate(axes):
                ax.plot(data[i]['val'], c=colors[0], label="val")
                ax.plot(data[i]['train'], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")
                
            plt.tight_layout()
            plt.show()
            
    def load_pretrained_model(self, state_path: str):
        """Cargar modelo preentrenado"""
        self.net.load_state_dict(torch.load(state_path))
        print("Modelo preentrenado cargado")
        
    def load_training_logs(self, logs_path: str):
        """Cargar logs de entrenamiento previo"""
        train_logs = pd.read_csv(logs_path)
        self.losses["train"] = train_logs.loc[:, "train_loss"].to_list()
        self.losses["val"] = train_logs.loc[:, "val_loss"].to_list()
        self.dice_scores["train"] = train_logs.loc[:, "train_dice"].to_list()
        self.dice_scores["val"] = train_logs.loc[:, "val_dice"].to_list()
        self.jaccard_scores["train"] = train_logs.loc[:, "train_jaccard"].to_list()
        self.jaccard_scores["val"] = train_logs.loc[:, "val_jaccard"].to_list()
        print("Logs de entrenamiento cargados")
    
    def _save_train_history(self):
        """Guardar modelo y logs de entrenamiento"""
        # Guardar modelo final
        torch.save(self.net.state_dict(), "last_epoch_model.pth")
        
        # Guardar logs
        logs_ = [self.losses, self.dice_scores, self.jaccard_scores]
        log_names_ = ["_loss", "_dice", "_jaccard"]
        logs = [logs_[i][key] for i in range(len(logs_)) for key in logs_[i]]
        log_names = [key + log_names_[i] for i in range(len(logs_)) for key in logs_[i]]
        
        pd.DataFrame(dict(zip(log_names, logs))).to_csv("train_log.csv", index=False)
        print("Modelo y logs guardados")
