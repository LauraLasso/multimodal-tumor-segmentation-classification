# -*- coding: utf-8 -*-
"""
Módulo para cargar modelos entrenados
"""
import torch
import torch.nn as nn
 
from models.LViT.nets.LViT import LViT
import models.LViT.Config as config

from pathlib import Path
from typing import Optional

def load_trained_model(
    model_path: str,
    model_type: str = 'LViT',
    device: str = 'cuda',
    use_data_parallel: bool = True
) -> torch.nn.Module:
    """
    Cargar un modelo entrenado desde checkpoint
    
    Args:
        model_path: Ruta al archivo del modelo (.pth.tar)
        model_type: Tipo de modelo ('LViT')
        device: Dispositivo ('cuda' o 'cpu')
        use_data_parallel: Si usar DataParallel para múltiples GPUs
    
    Returns:
        Modelo cargado y configurado
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    # Configuración del modelo
    config_vit = config.get_CTranS_config()
    model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    model = model.to(device)

    # Configurar múltiples GPUs si están disponibles
    if use_data_parallel and torch.cuda.device_count() > 1:
        print(f"Usando múltiples GPUs: {torch.cuda.device_count()}")
        model = nn.DataParallel(model)

    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    print(f"Modelo {model_type} cargado correctamente desde:\n{model_path}")
    return model

def setup_reproducibility(seed: int = 770):
    """
    Configurar semillas para reproducibilidad
    
    Args:
        seed: Semilla para reproducibilidad
    """
    import random
    import numpy as np
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def diagnose_model(model, sample_batch, device="cuda"):
    """
    Diagnosticar el modelo con un batch de ejemplo
    """
    model.eval()
    with torch.no_grad():
        x = sample_batch['image'].to(device)
        text = sample_batch['text'].to(device)
        age = sample_batch['age'].to(device)
        
        print("=== DIAGNÓSTICO DEL MODELO ===")
        print(f"Dimensiones de entrada:")
        print(f"  - Imagen: {x.shape}")
        print(f"  - Texto: {text.shape}")
        print(f"  - Edad: {age.shape}")
        
        try:
            outputs = model(x, text, age)
            if isinstance(outputs, tuple):
                print(f"El modelo devuelve {len(outputs)} outputs:")
                for i, output in enumerate(outputs):
                    print(f"  - Output {i}: {output.shape}")
            else:
                print(f"El modelo devuelve un solo output: {outputs.shape}")
                
        except Exception as e:
            print(f"Error en el forward pass: {e}")
            
            # Intentar con diferentes configuraciones
            try:
                # Probar solo con imagen
                output_img = model.encoder(x)
                print(f"Solo encoder de imagen funciona: {output_img.shape}")
            except:
                print("Error también con solo encoder de imagen")
