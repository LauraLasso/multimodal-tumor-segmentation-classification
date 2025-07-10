# -*- coding: utf-8 -*-
"""
Utilidades generales
"""
import torch
import numpy as np
import random
from pathlib import Path
from typing import Union

def set_seed(seed: int):
    """Configurar semilla global"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def count_parameters(model: torch.nn.Module) -> int:
    """Contar parámetros del modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model: torch.nn.Module, optimizer, epoch: int, loss: float, 
                   filepath: Union[str, Path]):
    """Guardar checkpoint completo"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath: Union[str, Path], model: torch.nn.Module, 
                   optimizer=None) -> dict:
    """Cargar checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss']
    }

def get_model_summary(model: torch.nn.Module, input_size: tuple):
    """Obtener resumen del modelo"""
    total_params = count_parameters(model)
    print(f"Total de parámetros entrenables: {total_params:,}")
    print(f"Tamaño de entrada esperado: {input_size}")
    
    # Crear tensor de prueba
    dummy_input = torch.randn(1, *input_size)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Tamaño de salida: {output.shape}")
    except Exception as e:
        print(f"Error al calcular salida: {e}")

def create_directories(paths: list):
    """Crear directorios si no existen"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

class EarlyStopping:
    """Early stopping para entrenamiento"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model: torch.nn.Module):
        """Guardar mejores pesos"""
        self.best_weights = model.state_dict().copy()
