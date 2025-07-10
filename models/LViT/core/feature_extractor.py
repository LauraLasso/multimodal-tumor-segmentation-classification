# -*- coding: utf-8 -*-
"""
Módulo para extracción de características latentes
"""
import torch
import pandas as pd
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path

def extract_latent_features_to_df(
    model: torch.nn.Module,
    dataloader: DataLoader,
    set_name: str,  # NUEVO PARÁMETRO
    device: str = "cuda",
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Extraer características latentes del modelo y convertir a DataFrame
    
    Args:
        model: Modelo entrenado
        dataloader: DataLoader con los datos
        set_name: Nombre del conjunto ("Train_Folder", "Val_Folder", "Test_Folder")
        device: Dispositivo de cómputo
        save_path: Ruta para guardar CSV (opcional)
    
    Returns:
        DataFrame con características latentes, ID, edad, días de supervivencia y set
    """
    model.eval()
    feature_rows = []
    id_list = []
    age_list = []
    sd_list = []

    with torch.no_grad():
        for i, (batch, names) in enumerate(dataloader, 1):
            try:
                x = batch['image'].to(device)
                text = batch['text'].to(device)
                age = batch['age'].to(device)
                survival_days = batch['target'].to(device)

                # Inferencia
                outputs = model(x, text, age)
                
                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    y4_encoder = outputs[2]
                else:
                    y4_encoder = outputs[-1] if isinstance(outputs, tuple) else outputs
                
                # Pooling
                if len(y4_encoder.shape) == 3:
                    pooled_flat = y4_encoder.mean(dim=1)
                elif len(y4_encoder.shape) == 2:
                    pooled_flat = y4_encoder
                else:
                    pooled_flat = y4_encoder.view(y4_encoder.size(0), -1)

                # Extraer características para cada muestra del batch
                for j in range(pooled_flat.size(0)):
                    features = pooled_flat[j].cpu().numpy()
                    feature_rows.append(features)
                    id_list.append(names[j])
                    age_list.append(age[j].item())
                    sd_list.append(survival_days[j].item())
                    
            except Exception as e:
                print(f"Error en batch {i}: {e}")
                continue

    if not feature_rows:
        raise ValueError("No se pudieron extraer características de ningún batch")

    # Crear DataFrame con columnas latentes + metadatos + set
    num_features = feature_rows[0].shape[0]
    columns = [f"latent_f{i}" for i in range(num_features)]
    df = pd.DataFrame(feature_rows, columns=columns)
    df.insert(0, "Id", id_list)
    df["Age"] = age_list
    df["Survival_Days"] = sd_list
    df["set"] = set_name  # AÑADIR COLUMNA SET

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"CSV guardado en: {save_path}")

    return df

def extract_features_from_all_sets(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    save_dir: str = "./features/",
    device: str = "cuda"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extraer características de todos los conjuntos con identificadores de set
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extraer características de cada conjunto CON IDENTIFICADOR
    train_df = extract_latent_features_to_df(
        model, train_loader, "Train_Folder", device, 
        save_path=save_dir / "features_train.csv"
    )
    
    val_df = extract_latent_features_to_df(
        model, val_loader, "Val_Folder", device,
        save_path=save_dir / "features_val.csv"
    )
    
    test_df = extract_latent_features_to_df(
        model, test_loader, "Test_Folder", device,
        save_path=save_dir / "features_test.csv"
    )
    
    return train_df, val_df, test_df

def verify_and_clean_data(df):
    """
    Verificar y limpiar datos problemáticos
    """
    print("=== Verificación de datos ===")
    print(f"Total de muestras: {len(df)}")
    print(f"Valores NaN en 'set': {df['set'].isna().sum()}")
    print(f"Valores únicos en 'set': {df['set'].unique()}")
    
    # Eliminar filas con set NaN
    df_clean = df.dropna(subset=['set'])
    print(f"Muestras después de limpiar NaN: {len(df_clean)}")
    
    return df_clean
