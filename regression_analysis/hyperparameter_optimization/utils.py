# -*- coding: utf-8 -*-
"""
Utilidades generales para análisis de regresión
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

class Config:
    """Configuración global"""
    seed = 42

def filter_negative_predictions(
    df: pd.DataFrame, 
    target: str, 
    pred_target: str
) -> pd.DataFrame:
    """
    Filtra predicciones negativas de un DataFrame
    
    Args:
        df: DataFrame con predicciones
        target: Nombre de la columna target
        pred_target: Nombre de la columna de predicciones
    
    Returns:
        DataFrame filtrado sin valores negativos
    """
    mask = (df[target] >= 0) & (df[pred_target] >= 0)
    return df[mask]

def prepare_target_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    target: str,
    features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepara los datos eliminando valores nulos para un target específico
    
    Args:
        train_df, val_df, test_df: DataFrames originales
        target: Nombre del target
        features: Lista de características
    
    Returns:
        Tuple con DataFrames limpios
    """
    train_clean = train_df[[target] + features].dropna()
    val_clean = val_df[[target] + features].dropna()
    test_clean = test_df[[target] + features].dropna()
    
    return train_clean, val_clean, test_clean

def get_desnormalization_params(target: str) -> Tuple[float, float]:
    """
    Obtiene los parámetros de desnormalización para un target específico
    
    Args:
        target: Nombre del target
    
    Returns:
        Tuple con (min_val_real, max_val_real)
    """
    if "age" in target.lower():
        return 18.975, 86.652
    else:  # survival days
        return 5.0, 1767.0

def setup_pandas_display():
    """
    Configura pandas para mostrar tablas completas
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

def print_section_header(title: str, char: str = "=", width: int = 80):
    """
    Imprime un encabezado de sección formateado
    
    Args:
        title: Título de la sección
        char: Carácter para el borde
        width: Ancho total del encabezado
    """
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")
