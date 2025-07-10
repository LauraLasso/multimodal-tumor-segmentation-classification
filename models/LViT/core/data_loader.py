# -*- coding: utf-8 -*-
"""
Módulo para configuración y carga de datos
"""
import torch
from torch.utils.data import DataLoader
 
from models.LViT.Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D
from models.LViT.utils import read_text, read_text_brats2020
from torchvision import transforms
import time
import os
import random
from typing import Tuple, Dict, Any

class DataConfig:
    """Configuración centralizada para datos"""
    def __init__(self):
        # === GENERAL SETTINGS ===
        self.seed = 770
        self.batch_size = 2
        self.img_size = 224
        self.n_channels = 3
        self.n_labels = 1
        
        # === TASK & DATA ===
        self.task_name = 'Processed_BraTS2020'
        self.model_name = 'LViT'
        
        # === PATHS ===
        self.base_path = f'./datasets/{self.task_name}'
        self.train_dataset = f'{self.base_path}/Train_Folder/'
        self.val_dataset = f'{self.base_path}/Val_Folder/'
        self.test_dataset = f'{self.base_path}/Test_Folder/'

def worker_init_fn(worker_id: int, seed: int = 770):
    """Función de inicialización para workers del DataLoader"""
    random.seed(seed + worker_id)

def get_dataloaders(
    config: DataConfig = None,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crear DataLoaders para train, validation y test
    
    Args:
        config: Configuración de datos
        num_workers: Número de workers para DataLoader
        pin_memory: Si usar pin_memory
    
    Returns:
        Tuple con train_loader, val_loader, test_loader
    """
    if config is None:
        config = DataConfig()
    
    # Configurar entorno
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    
    # Transforms
    train_tf = transforms.Compose([
        RandomGenerator(output_size=[config.img_size, config.img_size])
    ])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    
    # Leer archivos de texto
    if config.task_name == 'Processed_BraTS2020':
        train_text = read_text_brats2020(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text_brats2020(config.val_dataset + 'Val_text.xlsx')
        test_text = read_text_brats2020(config.test_dataset + 'Test_text.xlsx')
    else:
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        test_text = read_text(config.test_dataset + 'Test_text.xlsx')
    
    # Crear datasets
    train_dataset = ImageToImage2D(
        config.train_dataset, config.task_name, train_text, 
        train_tf, image_size=config.img_size
    )
    val_dataset = ImageToImage2D(
        config.val_dataset, config.task_name, val_text, 
        val_tf, image_size=config.img_size
    )
    test_dataset = ImageToImage2D(
        config.test_dataset, config.task_name, test_text, 
        val_tf, image_size=config.img_size
    )
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, config.seed),
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, config.seed),
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, config.seed),
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
