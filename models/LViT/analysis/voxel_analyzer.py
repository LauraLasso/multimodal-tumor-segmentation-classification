# -*- coding: utf-8 -*-
"""
Módulo para análisis de estadísticas de vóxeles de máscaras
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import cv2
from typing import Dict, List, Optional
from pathlib import Path

def safe_skew(data: np.ndarray) -> float:
    """
    Calcular skewness de forma segura
    
    Args:
        data: Array de datos
    
    Returns:
        Valor de skewness o 0.0 si no se puede calcular
    """
    if len(data) < 2 or np.std(data) == 0:
        return 0.0
    return stats.skew(data)

def extract_voxel_stats_from_mask(
    mask_path: str, 
    patient_id: str,
    include_advanced_stats: bool = False
) -> Optional[Dict]:
    """
    Extraer estadísticas de vóxeles de una máscara
    
    Args:
        mask_path: Ruta a la máscara
        patient_id: ID del paciente
        include_advanced_stats: Si incluir estadísticas avanzadas
    
    Returns:
        Diccionario con estadísticas o None si hay error
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"No se pudo leer {mask_path}")
        return None

    mask = (mask > 0).astype(np.uint8)
    flat = mask.flatten().astype(np.float32)

    if flat.sum() == 0:
        return None  # máscara vacía

    stats_dict = {"Id": patient_id}
    
    if include_advanced_stats:
        # Estadísticas básicas
        stats_dict["mask_mean"] = flat.mean()
        stats_dict["mask_std"] = flat.std()
        stats_dict["mask_skew"] = stats.skew(flat)
        stats_dict["mask_kurtosis"] = stats.kurtosis(flat)

        # Análisis por intensidad
        mean = flat.mean()
        std = flat.std()
        intensive = flat[flat > mean]
        non_intensive = flat[flat < mean]
        more_intensive = flat[flat > mean + std]

        stats_dict["intensive_voxel_count"] = len(intensive)
        stats_dict["non_intensive_voxel_count"] = len(non_intensive)
        stats_dict["more_intensive_voxel_count"] = len(more_intensive)
        stats_dict["intensive_skew"] = safe_skew(intensive)
        stats_dict["non_intensive_skew"] = safe_skew(non_intensive)
        stats_dict["diff_skew_intensive"] = stats_dict["mask_skew"] - stats_dict["intensive_skew"]
        stats_dict["diff_skew_non_intensive"] = stats_dict["mask_skew"] - stats_dict["non_intensive_skew"]

        # Particiones locales
        for i, part in enumerate(np.array_split(flat, 10)):
            stats_dict[f"part{i}_mean"] = part.mean()

    return stats_dict

def generate_voxel_stats_from_folders(
    df: pd.DataFrame,
    base_path: str = "./datasets/Processed_BraTS2020",
    sets: List[str] = None,
    include_advanced_stats: bool = False
) -> pd.DataFrame:
    """
    Generar estadísticas de vóxeles para todos los conjuntos
    """
    if sets is None:
        sets = ["Train_Folder", "Val_Folder", "Test_Folder"]
    
    voxel_stats_all = []
    base_path = Path(base_path)

    for split in sets:
        mask_dir = base_path / split / "labelcol"
        
        if not mask_dir.exists():
            print(f"Directorio no encontrado: {mask_dir}")
            continue
            
        for fname in os.listdir(mask_dir):
            if fname.endswith(".png"):
                file_id = fname.replace("_seg.png", "").replace(".png", "")
                full_path = mask_dir / fname
                
                voxel_stats = extract_voxel_stats_from_mask(
                    str(full_path), file_id, include_advanced_stats
                )
                
                if voxel_stats:
                    # NO añadir columna 'set' aquí, se preservará del DataFrame original
                    voxel_stats_all.append(voxel_stats)

    # Crear DataFrame con estadísticas de vóxeles (SIN columna 'set')
    df_voxel = pd.DataFrame(voxel_stats_all)
    
    if df_voxel.empty:
        print("No se encontraron estadísticas de vóxeles")
        return df
    
    # SOLUCIÓN: Hacer merge preservando todas las columnas del DataFrame original
    df_merged = df.merge(df_voxel, on="Id", how="left")
    
    # Verificar que la columna 'set' se preservó
    if 'set' not in df_merged.columns:
        print("ADVERTENCIA: Columna 'set' perdida durante el merge")
        # Restaurar la columna 'set' del DataFrame original
        df_merged['set'] = df['set']
    
    print(f"Estadísticas de vóxeles añadidas para {len(df_voxel)} muestras")
    print(f"Columnas después del merge: {df_merged.columns.tolist()}")
    
    return df_merged
