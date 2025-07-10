import pandas as pd
import numpy as np
import os

def prepare_data_splits(csv_path, data_list):
    """
    Prepara las divisiones de datos para entrenamiento, validación y prueba
    """
    # Leer el CSV clínico que ya contiene la columna 'fold'
    df = pd.read_csv(csv_path)
    
    # Convertir lista de diccionarios con rutas en DataFrame
    df_paths = pd.DataFrame(data_list)
    
    # Asegurarse de que el campo que los une es 'Brats20ID'
    df_paths["Brats20ID"] = df_paths["subject"]
    
    # Unir datos clínicos y rutas
    df_merged = pd.merge(df_paths, df, on="Brats20ID", how="left")
    
    # Inicializar listas
    train_files = []
    val_files = []
    test_files = []
    
    # Clasificar pacientes según criterio
    for _, row in df_merged.iterrows():
        paciente = {
            "t1": row["t1"],
            "t1ce": row["t1ce"],
            "t2": row["t2"],
            "flair": row["flair"],
            "seg": row["seg"],
            "subject": row["subject"]
        }
        
        fold = row["fold"]
        edad = row["Age"]
        
        if fold == 0:
            val_files.append(paciente)
        elif pd.isna(edad):
            test_files.append(paciente)
        else:
            train_files.append(paciente)
    
    print(f"Training: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")
    
    return train_files, val_files, test_files

def get_brats_data(root_dir):
    all_images = []
    
    subject_dirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                          if os.path.isdir(os.path.join(root_dir, d))])
    
    for subject_dir in subject_dirs:
        subject = os.path.basename(subject_dir)
        
        # Find the four modalities and the segmentation mask
        flair = os.path.join(subject_dir, f"{subject}_flair.nii")
        t1 = os.path.join(subject_dir, f"{subject}_t1.nii")
        t1ce = os.path.join(subject_dir, f"{subject}_t1ce.nii")
        t2 = os.path.join(subject_dir, f"{subject}_t2.nii")
        seg = os.path.join(subject_dir, f"{subject}_seg.nii")
        
        files = [flair, t1, t1ce, t2, seg]
        if all(os.path.exists(f) for f in files):
            all_images.append({
                "flair": flair,
                "t1": t1,
                "t1ce": t1ce,
                "t2": t2,
                "seg": seg,
                "subject": subject
            })
    
    return all_images
