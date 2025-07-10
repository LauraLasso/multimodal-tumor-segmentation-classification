# -*- coding: utf-8 -*-
"""
Dataset y DataLoader para 3D U-Net
"""
import os
import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from skimage.transform import resize
from albumentations import Compose
from typing import Dict, Any, Optional
from pathlib import Path


class BratsDataset(Dataset):
    """Dataset para datos BraTS 2020"""
    
    def __init__(self, df: pd.DataFrame, phase: str = "test", is_resize: bool = False, target_size: tuple = (78, 120, 120)):
        self.df = df
        self.phase = phase
        self.augmentations = self.get_augmentations(phase)
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        self.is_resize = is_resize
        self.target_size = target_size
     
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        try:
            id_ = self.df.loc[idx, 'Brats20ID']
            root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
            
            # Cargar todas las modalidades
            images = []
            for data_type in self.data_types:
                img_path = os.path.join(root_path, id_ + data_type)
                img = self.load_img(img_path)
                
                if self.is_resize:
                    img = self.resize(img)
                
                img = self.normalize(img)
                images.append(img)
            
            img = np.stack(images)  # (4, D, H, W)
            
            if self.phase != "test":
                mask_path = os.path.join(root_path, id_ + "_seg.nii")
                mask = self.load_img(mask_path)
                
                if self.is_resize:
                    mask = self.resize(mask)
                
                mask = self.preprocess_mask_labels(mask)

                if img.shape[1:] != mask.shape[1:]:
                    raise ValueError(f"Shape mismatch en {id_}: img {img.shape}, mask {mask.shape}")
            
                if self.augmentations:
                    if img.shape[1:] == mask.shape[1:]:
                        augmented = self.augmentations(
                            image=img.astype(np.float32),
                            mask=mask.astype(np.float32)
                        )
                        img = augmented['image']
                        mask = augmented['mask']

                return {
                    "Id": id_,
                    "image": img,
                    "mask": mask,
                }
            
            return {
                "Id": id_,
                "image": img,
            }
            
        except Exception as e:
            print(f"Error en sample {idx}: {e}")
            raise e
        
    def load_img(self, file_path: str) -> np.ndarray:
        """Cargar imagen NIfTI"""
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalizar datos al rango [0, 1]"""
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)
    
    def resize(self, data: np.ndarray) -> np.ndarray:
        """Redimensionar datos"""
        return resize(data, self.target_size, preserve_range=True)

    def preprocess_mask_labels(self, mask: np.ndarray) -> np.ndarray:
        """Convertir máscara a 3 canales: WT, TC, ET"""
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        return np.stack([mask_WT, mask_TC, mask_ET])

    def get_augmentations(self, phase: str):
        """Obtener augmentaciones según la fase"""
        list_transforms = []
        # Aquí puedes añadir augmentaciones específicas
        return Compose(list_transforms)

def create_data_splits(config):
    """Crear splits de datos con estratificación por edad - CORREGIDO"""
    survival_info_df = pd.read_csv(os.path.join(config.train_root_dir, 'survival_info.csv'))
    name_mapping_df = pd.read_csv(os.path.join(config.train_root_dir, 'name_mapping.csv'))
    name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True)
    
    df = survival_info_df.merge(name_mapping_df, on="Brats20ID", how="right")
    
    # Crear rutas
    paths = []
    for _, row in df.iterrows():
        id_ = row['Brats20ID']
        phase = id_.split("_")[-2]
        
        if phase == 'Training':
            path = os.path.join(config.train_root_dir, id_)
        else:
            path = os.path.join(config.test_root_dir, id_)
        paths.append(path)
    
    df['path'] = paths
    
    # Filtrar y estratificar
    train_data = df.loc[df['Age'].notnull()].reset_index(drop=True)
    train_data["Age_rank"] = train_data["Age"] // 10 * 10
    train_data = train_data.loc[train_data['Brats20ID'] != 'BraTS20_Training_355'].reset_index(drop=True)
    
    skf = StratifiedKFold(n_splits=7, random_state=config.seed, shuffle=True)
    
    # Asignar folds
    for i, (train_index, val_index) in enumerate(skf.split(train_data, train_data["Age_rank"])):
        train_data.loc[val_index, "fold"] = i
    
    # CORRECCIÓN: Asignar fold a datos de test
    test_data = df.loc[~df['Age'].notnull()].reset_index(drop=True)
    test_data['fold'] = -1  # Fold especial para test
    
    # CORRECCIÓN: Devolver todos los datos juntos, no separados
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    
    # Para compatibilidad, también devolver los splits separados
    train_df = train_data.loc[train_data['fold'] != 0].reset_index(drop=True)
    val_df = train_data.loc[train_data['fold'] == 0].reset_index(drop=True)
    test_df = test_data.copy()

    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    
    return all_data, train_df, val_df, test_df

def get_dataloader(dataset: Dataset, path_to_csv: str, phase: str, fold: int = 0, 
                  batch_size: int = 1, num_workers: int = 0) -> DataLoader:
    """Crear DataLoader con verificaciones mejoradas"""
    print(f"\n--- Creando DataLoader para {phase} ---")
    print(f"CSV path: {path_to_csv}")
    print(f"Fold: {fold}")
    
    df = pd.read_csv(path_to_csv)
    print(f"Total filas en CSV: {len(df)}")
    
    if len(df) == 0:
        raise ValueError(f"El archivo CSV está vacío: {path_to_csv}")
    
    if 'fold' not in df.columns:
        raise ValueError("La columna 'fold' no existe en el CSV")
    
    print(f"Folds disponibles: {df['fold'].unique()}")
    
    train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
    val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

    df_phase = train_df if phase == "train" else val_df
    
    print(f"Datos para {phase}: {len(df_phase)} muestras")
    
    if len(df_phase) == 0:
        print(f"ERROR: No hay datos para la fase '{phase}' con fold={fold}")
        print(f"Distribución de folds: {df['fold'].value_counts().sort_index()}")
        raise ValueError(f"No hay datos para la fase '{phase}' con fold={fold}")
    
    dataset_instance = dataset(df_phase, phase)
    
    dataloader = DataLoader(
        dataset_instance,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True if phase == "train" else False,   
    )
    
    print(f"DataLoader creado: {len(dataloader)} batches")
    return dataloader

def get_test_dataloader(dataset_class, path_to_csv, root_folder, batch_size=1, num_workers=0):
    df_all = pd.read_csv(path_to_csv)
    
    all_patients = [p.name for p in Path(root_folder).iterdir() if p.is_dir()]
    csv_patients = set(df_all["Brats20ID"].astype(str).tolist())

    patients_missing = [p for p in all_patients if p not in csv_patients]

    df_test_from_csv = df_all[(df_all["fold"] != 0) & (df_all["Age"].isna())]

    test_ids = set(patients_missing + df_test_from_csv["Brats20ID"].astype(str).tolist())
    test_ids.discard("BraTS20_Training_355") 

    df_test = pd.DataFrame({"Brats20ID": list(test_ids)})
    df_test["path"] = df_test["Brats20ID"].apply(lambda x: os.path.join(root_folder, x))

    dataset = dataset_class(df_test, phase="valid")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False
    )
    return dataloader

