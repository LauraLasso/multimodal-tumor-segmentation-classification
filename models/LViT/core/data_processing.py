# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import cv2
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class BraTSDataProcessor:
    def __init__(self, dataset_dir, survival_csv, grade_csv, output_dir, img_size=224):
        self.dataset_dir = Path(dataset_dir)
        self.survival_csv = Path(survival_csv)
        self.grade_csv = Path(grade_csv)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        
        # Cargar datos clínicos
        self.df_survival = pd.read_csv(self.survival_csv)
        self.df_grade = pd.read_csv(self.grade_csv)
        
        # Calcular medias para valores faltantes
        self.mean_age = self.df_survival["Age"].dropna().astype(str).str.extract(r"(\d+)").astype(float).mean().values[0]
        self.mean_survival = self.df_survival["Survival_days"].dropna().astype(str).str.extract(r"(\d+)").astype(float).mean().values[0]
    
    def create_output_dirs(self, splits=["Train_Folder", "Val_Folder", "Test_Folder"]):
        """Crear estructura de directorios de salida"""
        for split in splits:
            (self.output_dir / split / "img").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labelcol").mkdir(parents=True, exist_ok=True)
    
    def normalize_image(self, img):
        """Normalizar imagen a rango 0-255"""
        if np.max(img) > np.min(img):
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        return img.astype(np.uint8)
    
    def save_middle_slice(self, img_nii, seg_nii, out_img_path, out_seg_path):
        """Guardar el corte medio con mayor segmentación"""
        img = img_nii.get_fdata()
        seg = seg_nii.get_fdata()
        z_sums = [np.sum(seg[:, :, z]) for z in range(seg.shape[2])]
        z_index = np.argmax(z_sums) if np.max(z_sums) > 0 else seg.shape[2] // 2
        
        img_slice = self.normalize_image(img[:, :, z_index])
        seg_slice = (seg[:, :, z_index] > 0).astype(np.uint8) * 255
        
        img_resized = cv2.resize(img_slice, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        seg_resized = cv2.resize(seg_slice, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(str(out_img_path), img_resized)
        cv2.imwrite(str(out_seg_path), seg_resized)
    
    def leer_nombres(self, txt_path):
        """Leer nombres de pacientes desde archivo TXT"""
        with open(txt_path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    
    def split_patients_from_txt(self, train_txt, val_txt, test_txt):
        """Dividir pacientes usando archivos TXT"""
        train_ids = self.leer_nombres(train_txt)
        val_ids = self.leer_nombres(val_txt)
        test_ids = self.leer_nombres(test_txt)
        
        patients = sorted([d for d in self.dataset_dir.iterdir() if d.is_dir()])
        patients_dict = {p.name: p for p in patients}
        
        train = [patients_dict[pid] for pid in train_ids if pid in patients_dict]
        val = [patients_dict[pid] for pid in val_ids if pid in patients_dict]
        test = [patients_dict[pid] for pid in test_ids if pid in patients_dict]
        
        return {
            "Train_Folder": train,
            "Val_Folder": val,
            "Test_Folder": test
        }
    
    def split_patients_train_test_split(self, test_size=0.1, val_size=0.1111, random_state=42):
        """Dividir pacientes usando train_test_split"""
        patients = sorted([d for d in self.dataset_dir.iterdir() if d.is_dir()])
        train_val, test = train_test_split(patients, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train_val, test_size=val_size, random_state=random_state)
        
        return {
            "Train_Folder": train,
            "Val_Folder": val,
            "Test_Folder": test
        }
    
    def split_patients_from_csv(self, partition_csv):
        """Dividir pacientes usando CSV de partición por estratificación"""
        df_split = pd.read_csv(partition_csv)
        patients_dir = {p.name: p for p in self.dataset_dir.iterdir() if p.is_dir()}
        train, val, test = [], [], []
        
        pids_in_csv = set(df_split["Brats20ID"].astype(str))
        
        for _, row in df_split.iterrows():
            pid = str(row["Brats20ID"])
            fold = row["fold"]
            age = row.get("Age", None)
            
            if pid not in patients_dir:
                continue
            
            if fold == 0:
                val.append(patients_dir[pid])
            elif pd.notna(age) and str(age).strip() != "":
                train.append(patients_dir[pid])
            else:
                test.append(patients_dir[pid])
        
        # Agregar pacientes no listados en CSV a test
        for pid, path in patients_dir.items():
            if pid not in pids_in_csv:
                test.append(path)
        
        return {
            "Train_Folder": train,
            "Val_Folder": val,
            "Test_Folder": test
        }
    
    def extract_clinical_data(self, pid, fill_missing_for_description=False):
        """Extraer datos clínicos de un paciente"""
        row_survival = self.df_survival[self.df_survival["Brats20ID"] == pid]
        row_grade = self.df_grade[self.df_grade["BraTS_2020_subject_ID"] == pid]
        
        # Edad - distinguir entre valor real y valor para descripción
        age_for_data = None  # Para columnas del Excel
        age_for_description = None  # Para texto descriptivo
        
        try:
            raw_age = row_survival["Age"].values[0]
            if pd.notna(raw_age):
                age_extracted = float(re.search(r"\d+", str(raw_age)).group())
                age_for_data = age_extracted
                age_for_description = age_extracted
        except:
            pass
        
        # Si no hay edad real, usar media solo para datos, no para descripción
        if age_for_data is None and fill_missing_for_description:
            age_for_data = self.mean_age
        
        # Supervivencia - mismo enfoque
        survival_for_data = None
        survival_for_description = None
        
        try:
            survival_str = str(row_survival["Survival_days"].values[0])
            if pd.notna(row_survival["Survival_days"].values[0]):
                match = re.search(r"(\d+)", survival_str)
                if match:
                    survival_extracted = float(match.group(1))
                    survival_for_data = survival_extracted
                    survival_for_description = survival_extracted
        except:
            pass
        
        # Si no hay supervivencia real, usar media solo para datos, no para descripción
        if survival_for_data is None and fill_missing_for_description:
            survival_for_data = self.mean_survival
        
        # Grado y resección (sin cambios)
        grade = row_grade["Grade"].values[0] if not row_grade.empty and pd.notna(row_grade["Grade"].values[0]) else None
        extent = row_survival["Extent_of_Resection"].values[0] if not row_survival.empty and pd.notna(row_survival["Extent_of_Resection"].values[0]) else None
        
        return {
            "age": age_for_data,
            "age_for_description": age_for_description,
            "survival": survival_for_data,
            "survival_for_description": survival_for_description,
            "grade": grade,
            "extent": extent,
            "row_survival": row_survival,
            "row_grade": row_grade
        }
    
    def build_description(self, clinical_data, description_type="age"):
        """Construir descripción según el tipo especificado - solo con datos reales"""
        description_parts = []
        
        # Solo agregar edad si existe el dato real
        if description_type == "age" and clinical_data["age_for_description"] is not None:
            description_parts.append(f"Edad del paciente: {clinical_data['age_for_description']:.1f} años.")
        # Solo agregar supervivencia si existe el dato real
        elif description_type == "survival" and clinical_data["survival_for_description"] is not None:
            description_parts.append(f"Días de supervivencia estimados: {clinical_data['survival_for_description']:.0f} días.")
        
        # Grado y resección solo si existen
        if clinical_data["grade"]:
            grade_desc = "alto grado" if clinical_data["grade"] == "HGG" else "bajo grado"
            description_parts.append(f"Tipo de glioma: {grade_desc} ({clinical_data['grade']}).")
        
        if clinical_data["extent"]:
            description_parts.append(f"Grado de resección quirúrgica: {clinical_data['extent']}.")
        
        return " ".join(description_parts)
    
    def process_patients(self, split_map, description_type="age", skip_missing_data=False, fill_missing_for_data=True):
        """Procesar pacientes y generar datos XLSX"""
        xlsx_data = {k: [] for k in split_map}
        
        for split, patient_dirs in split_map.items():
            for p in patient_dirs:
                pid = p.name
                
                # Extraer datos clínicos
                clinical_data = self.extract_clinical_data(pid, fill_missing_for_description=fill_missing_for_data)
                
                # Validar datos si es necesario
                if skip_missing_data:
                    row_survival = clinical_data["row_survival"]
                    if (row_survival.empty or 
                        pd.isna(row_survival["Age"].values[0]) or 
                        pd.isna(row_survival["Survival_days"].values[0])):
                        print(f"Saltando {pid} por falta de edad o días de supervivencia.")
                        continue
                
                # Procesar imágenes (sin cambios)
                t1ce_file = p / f"{pid}_t1ce.nii"
                seg_file = p / "W39_1998.09.19_Segm.nii" if pid == "BraTS20_Training_355" else p / f"{pid}_seg.nii"
                
                img_out = self.output_dir / split / "img" / f"{pid}.png"
                seg_out = self.output_dir / split / "labelcol" / f"{pid}_seg.png"
                
                self.save_middle_slice(nib.load(t1ce_file), nib.load(seg_file), img_out, seg_out)
                
                # Construir descripción (solo con datos reales)
                custom_description = self.build_description(clinical_data, description_type)
                
                # Usar medias para columnas de datos si es necesario
                age_for_excel = clinical_data["age"] if clinical_data["age"] is not None else (self.mean_age if fill_missing_for_data else None)
                survival_for_excel = clinical_data["survival"] if clinical_data["survival"] is not None else (self.mean_survival if fill_missing_for_data else None)
                
                # Agregar entrada
                entry = {
                    "filename": f"{pid}_seg.png",
                    "description": custom_description,
                    "Age": age_for_excel,
                    "Survival_days": survival_for_excel
                }
                
                # Agregar campos adicionales si están disponibles
                if clinical_data["grade"]:
                    entry["Grade"] = clinical_data["grade"]
                if clinical_data["extent"]:
                    entry["Extent_of_Resection"] = clinical_data["extent"]
                
                xlsx_data[split].append(entry)
        
        return xlsx_data
    
    def save_xlsx_data(self, xlsx_data):
        """Guardar datos XLSX"""
        for split in xlsx_data:
            df = pd.DataFrame(xlsx_data[split])
            out_file = self.output_dir / split / f"info_{split.lower()}.xlsx"
            df.to_excel(out_file, index=False)
    
    # === NUEVOS MÉTODOS DE NORMALIZACIÓN ===
    
    def normalize_survival_days_only(self, splits=["Train_Folder", "Val_Folder", "Test_Folder"], 
                                   file_suffix="_text.xlsx", normalized_suffix="_text_normalized.xlsx"):
        """
        Normalizar únicamente los días de supervivencia usando MinMaxScaler
        """
        # Crear carpeta para archivos normalizados
        normalized_dir = self.output_dir / "Normalized_Texts"
        normalized_dir.mkdir(exist_ok=True)
        
        # Cargar todos los archivos y concatenar para normalizar en conjunto
        all_dfs = []
        for split in splits:
            file_path = self.output_dir / split / f"{split.split('_')[0]}{file_suffix}"
            if file_path.exists():
                df = pd.read_excel(file_path)
                df["split"] = split  # Guardar origen
                all_dfs.append(df)
            else:
                print(f"Archivo no encontrado: {file_path}")
        
        if not all_dfs:
            print("No se encontraron archivos para normalizar.")
            return
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Normalizar la columna 'Survival_days'
        scaler = MinMaxScaler()
        combined_df["Survival_days_normalized"] = scaler.fit_transform(combined_df[["Survival_days"]])
        
        # Guardar archivos divididos por split
        for split in splits:
            df_split = combined_df[combined_df["split"] == split].drop(columns=["split"])
            out_path = normalized_dir / f"{split.split('_')[0]}{normalized_suffix}"
            df_split.to_excel(out_path, index=False)
            print(f"Archivo normalizado guardado: {out_path}")
        
        return combined_df
    
    def normalize_age_and_survival(self, splits=["Train_Folder", "Val_Folder", "Test_Folder"], 
                                 file_suffix="_text.xlsx", normalized_suffix="_text_normalized.xlsx"):
        """
        Normalizar tanto la edad como los días de supervivencia usando MinMaxScaler
        """
        # Crear carpeta para archivos normalizados
        normalized_dir = self.output_dir / "Normalized_Texts"
        normalized_dir.mkdir(exist_ok=True)
        
        # Cargar todos los archivos y concatenar para normalizar en conjunto
        all_dfs = []
        for split in splits:
            file_path = self.output_dir / split / f"{split.split('_')[0]}{file_suffix}"
            if file_path.exists():
                df = pd.read_excel(file_path)
                df["split"] = split  # Guardar origen
                all_dfs.append(df)
            else:
                print(f"Archivo no encontrado: {file_path}")
        
        if not all_dfs:
            print("No se encontraron archivos para normalizar.")
            return
        
        # Unir todos los datos
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Aplicar normalización MinMax a ambas columnas: Age y Survival_days
        scaler = MinMaxScaler()
        combined_df[["Age_normalized", "Survival_days_normalized"]] = scaler.fit_transform(
            combined_df[["Age", "Survival_days"]]
        )
        
        # Guardar archivos divididos por split
        for split in splits:
            df_split = combined_df[combined_df["split"] == split].drop(columns=["split"])
            out_path = normalized_dir / f"{split.split('_')[0]}{normalized_suffix}"
            df_split.to_excel(out_path, index=False)
            print(f"Archivo normalizado guardado: {out_path}")
        
        return combined_df
    
    def rename_columns_for_regression(self, file_path, normalization_type="survival_only"):
        """
        Renombrar columnas para usar valores normalizados como principales
        
        Args:
            file_path: Ruta al archivo Excel
            normalization_type: "survival_only" o "both" (edad y supervivencia)
        """
        # Leer el Excel
        df = pd.read_excel(file_path)
        
        print("Columnas actuales:", df.columns.tolist())
        
        if normalization_type == "survival_only":
            # Renombrar solo supervivencia
            df.rename(columns={
                "Survival_days": "Normal_Survival_days",
                "Survival_days_normalized": "Survival_days"
            }, inplace=True)
        elif normalization_type == "both":
            # Renombrar edad y supervivencia
            df.rename(columns={
                "Age": "Normal_Age",
                "Age_normalized": "Age",
                "Survival_days": "Normal_Survival_days",
                "Survival_days_normalized": "Survival_days"
            }, inplace=True)
        
        # Guardar el archivo modificado
        df.to_excel(file_path, index=False)
        
        print("Nuevas columnas:", df.columns.tolist())
        print(f"Archivo actualizado: {file_path}")
        
        return df
    
    def apply_normalization_workflow(self, normalization_type="survival_only", 
                                   splits=["Train_Folder", "Val_Folder", "Test_Folder"],
                                   file_suffix="_text.xlsx", update_original_files=True):
        """
        Aplicar el flujo completo de normalización
        
        Args:
            normalization_type: "survival_only" o "both"
            splits: Lista de splits a procesar
            file_suffix: Sufijo de los archivos originales
            update_original_files: Si True, actualiza los archivos originales con columnas renombradas
        """
        print(f"Iniciando normalización: {normalization_type}")
        
        # Aplicar normalización
        if normalization_type == "survival_only":
            combined_df = self.normalize_survival_days_only(splits, file_suffix)
        elif normalization_type == "both":
            combined_df = self.normalize_age_and_survival(splits, file_suffix)
        else:
            raise ValueError("normalization_type debe ser 'survival_only' o 'both'")
        
        # Renombrar columnas en archivos originales si se solicita
        if update_original_files:
            for split in splits:
                file_path = self.output_dir / split / f"{split.split('_')[0]}{file_suffix}"
                if file_path.exists():
                    self.rename_columns_for_regression(file_path, normalization_type)
        
        print("Normalización completada.")
        return combined_df