# -*- coding: utf-8 -*-
"""
Módulo para aumento de datos
"""
import pandas as pd
import numpy as np
from typing import List, Optional

def augment_high_survival_samples(
    df: pd.DataFrame,
    survival_threshold: float = None,
    augmentations_per_sample: int = 7,
    noise_scale: float = 0.1,
    min_sd: float = 5,
    max_sd: float = 1767,
    survival_column: str = "Survival_Days"
) -> pd.DataFrame:
    """
    Aumentar muestras con alta supervivencia usando ruido gaussiano
    
    Args:
        df: DataFrame con características latentes
        survival_threshold: Umbral de supervivencia (se calcula automáticamente si es None)
        augmentations_per_sample: Número de aumentos por muestra
        noise_scale: Escala del ruido gaussiano
        min_sd: Valor mínimo de supervivencia para normalización
        max_sd: Valor máximo de supervivencia para normalización
        survival_column: Nombre de la columna de supervivencia
    
    Returns:
        DataFrame con muestras aumentadas
    """
    # Calcular umbral normalizado si no se proporciona
    if survival_threshold is None:
        survival_threshold = (750 - min_sd) / (max_sd - min_sd)
    
    # Asegurar que la columna esté en formato numérico
    df[survival_column] = pd.to_numeric(df[survival_column], errors="coerce")
    
    # Identificar columnas latentes
    latent_columns = [col for col in df.columns if col.startswith("latent_f")]
    
    if not latent_columns:
        raise ValueError("No se encontraron columnas latentes (que empiecen con 'latent_f')")
    
    # Filtrar pacientes con supervivencia alta
    high_sd = df[df[survival_column] > survival_threshold]
    print(f"Número de pacientes con SD > {survival_threshold:.3f} (normalizado): {len(high_sd)}")
    
    if high_sd.empty:
        print("No hay muestras para aumentar")
        return df
    
    # Generar muestras aumentadas
    augmented_rows = []
    
    for _, row in high_sd.iterrows():
        for aug_idx in range(augmentations_per_sample):
            new_row = row.copy()
            
            # Añadir ruido gaussiano a las características latentes
            noise = np.random.normal(
                loc=0, 
                scale=noise_scale, 
                size=len(latent_columns)
            )
            new_row[latent_columns] = new_row[latent_columns] + noise
            
            # Modificar ID para identificar como aumentado
            new_row["Id"] = f"{new_row['Id']}_aug{aug_idx}"
            augmented_rows.append(new_row)
    
    # Crear DataFrame con muestras aumentadas
    augmented_df = pd.DataFrame(augmented_rows)
    
    # Concatenar con el original
    df_augmented = pd.concat([df, augmented_df], ignore_index=True)
    
    print(f"Se añadieron {len(augmented_rows)} muestras aumentadas")
    return df_augmented

def split_augmented_data(
    df_augmented: pd.DataFrame,
    set_column: str = "set"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dividir datos aumentados por conjuntos
    
    Args:
        df_augmented: DataFrame con datos aumentados
        set_column: Nombre de la columna que indica el conjunto
    
    Returns:
        Tuple con DataFrames de train, val y test
    """
    train_df = df_augmented[df_augmented[set_column] == "Train_Folder"]
    val_df = df_augmented[df_augmented[set_column] == "Val_Folder"]
    test_df = df_augmented[df_augmented[set_column] == "Test_Folder"]
    
    print(f"Train: {len(train_df)} muestras")
    print(f"Val: {len(val_df)} muestras")
    print(f"Test: {len(test_df)} muestras")
    
    return train_df, val_df, test_df
