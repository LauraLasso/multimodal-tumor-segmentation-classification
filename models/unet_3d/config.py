# -*- coding: utf-8 -*-
"""
Configuraciones globales del proyecto 3D U-Net
"""
import os
import numpy as np
import torch
from pathlib import Path

class GlobalConfig:
    """Configuración centralizada para el proyecto"""
    
    # Rutas del dataset
    root_dir = './brats20-dataset-training-validation'
    train_root_dir = './brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = './brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    path_to_csv = './train_data.csv'
    
    # Rutas de modelos y logs
    pretrained_model_path = './brats20logs/brats2020logs/unet/last_epoch_model.pth'
    train_logs_path = './brats20logs/brats2020logs/unet/train_log.csv'
    ae_pretrained_model_path = './brats20logs/brats2020logs/ae/autoencoder_best_model.pth'
    tab_data = './brats20logs/brats2020logs/data/df_with_voxel_stats_and_latent_features.csv'
    
    # Parámetros del modelo
    seed = 55
    n_channels = 24
    in_channels = 4  # 4 modalidades MRI
    n_classes = 3    # WT, TC, ET
    
    # Parámetros de entrenamiento
    batch_size = 1
    num_epochs = 50
    learning_rate = 5e-4
    accumulation_steps = 4
    fold = 0
    
    # Parámetros de datos
    is_resize = False
    target_size = (78, 120, 120)

def seed_everything(seed: int):
    """Configurar semillas para reproducibilidad"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Obtener dispositivo disponible"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# Instancia global de configuración
config = GlobalConfig()
seed_everything(config.seed)
