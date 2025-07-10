# datasets.py
import nibabel as nib, numpy as np, pandas as pd, os
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose
import torch
from pathlib import Path

class AutoEncoderDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str = "test"):
        self.df = df
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        # load all modalities
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)

            img = self.normalize(img)
            images.append(img.astype(np.float32))
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
    
        
        return {
            "Id": id_,
            "data": img,
            "label": img,
            }
    
    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data
    
    def normalize(self, data: np.ndarray,  mean=0.0, std=1.0):
        """Normilize image value between 0 and 1."""
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

def get_augmentations(phase):
    list_transforms = []
    
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    path_to_csv: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 0,
):
    '''Returns: dataloader for the model training'''
    df = pd.read_csv(path_to_csv)
    
    train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
    val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

    df = train_df if phase == "train" else val_df
    dataset = dataset(df, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True,   
    )

    return dataloader

def get_test_dataloader(dataset_class, train_csv, root_folder, batch_size=1, num_workers=0):
    """
    Carga los pacientes del conjunto de test:
    - Pacientes que están en disco pero no en el CSV
    - O que están en el CSV con fold != 0 y sin edad (Age NaN)
    - Excluye explícitamente al paciente BraTS20_Training_355
    """
    # Leer el CSV original
    df_all = pd.read_csv(train_csv)
    csv_ids = set(df_all["Brats20ID"].astype(str).tolist())

    # Leer todos los nombres de carpetas en disco
    all_disk_ids = [p.name for p in Path(root_folder).iterdir() if p.is_dir()]

    # 1. Pacientes que no están en el CSV
    ids_not_in_csv = [pid for pid in all_disk_ids if pid not in csv_ids]

    # 2. Pacientes que están en el CSV pero cumplen condiciones de test
    df_test_extra = df_all[(df_all["fold"] != 0) & (df_all["Age"].isna())]
    ids_test_from_csv = df_test_extra["Brats20ID"].astype(str).tolist()

    # Unir y eliminar duplicados
    test_ids = list(set(ids_not_in_csv + ids_test_from_csv))

    # Eliminar explícitamente el paciente 355
    test_ids = [pid for pid in test_ids if pid != "BraTS20_Training_355"]

    # Crear dataframe con paths
    df_test = pd.DataFrame({"Brats20ID": test_ids})
    df_test["path"] = df_test["Brats20ID"].apply(lambda x: os.path.join(root_folder, x))

    # Crear dataset y dataloader
    dataset = dataset_class(df_test, phase="test") #, is_resize=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return dataloader