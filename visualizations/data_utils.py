import os
import glob
import pandas as pd

def obtener_todos_los_pacientes_brats(directorio_imagenes):
    """
    Función para obtener todos los IDs de pacientes del directorio de imágenes
    """
    # Buscar todas las carpetas de pacientes en el directorio
    patron_pacientes = os.path.join(directorio_imagenes, "BraTS20_Training_*")
    carpetas_pacientes = glob.glob(patron_pacientes)
    
    # Extraer solo los nombres de las carpetas (IDs de pacientes)
    ids_pacientes = [os.path.basename(carpeta) for carpeta in carpetas_pacientes]
    ids_pacientes.sort()  # Ordenar para consistencia
    
    return ids_pacientes

def crear_dataset_completo_brats(ruta_csv_survival, directorio_imagenes, valores_faltantes=None):
    """
    Función para crear un dataset completo incluyendo pacientes que no están en survival_info.csv
    """
    if valores_faltantes is None:
        valores_faltantes = ['', 'NA', 'N/A', 'NULL', '-', '?', 'Missing', 'No data']
    
    # Leer el archivo survival_info.csv
    df_survival = pd.read_csv(ruta_csv_survival, na_values=valores_faltantes, keep_default_na=True)
    
    # Obtener todos los pacientes del directorio de imágenes
    todos_los_pacientes = obtener_todos_los_pacientes_brats(directorio_imagenes)
    
    # Crear DataFrame completo con todos los pacientes
    df_completo = pd.DataFrame({'Brats20ID': todos_los_pacientes})
    
    # Hacer merge con los datos de supervivencia (left join para mantener todos los pacientes)
    df_final = df_completo.merge(df_survival, on='Brats20ID', how='left')
    
    print(f"Total de pacientes en el directorio: {len(todos_los_pacientes)}")
    print(f"Pacientes con datos de supervivencia: {len(df_survival)}")
    print(f"Pacientes sin datos de supervivencia: {len(todos_los_pacientes) - len(df_survival)}")
    
    return df_final, df_survival
