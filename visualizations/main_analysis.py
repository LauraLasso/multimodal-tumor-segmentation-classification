import os
from data_utils import crear_dataset_completo_brats
from visualizations_basic import visualizar_valores_faltantes_brats
from visualizations_advanced import (
    crear_grafico_dona_valores_faltantes,
    crear_heatmap_completitud,
    crear_grafico_area_apilada,
    crear_grafico_radar_completitud
)
from visualizations_3d import plot_segmentation_3D_with_shadows

def analizar_brats2020():
    """
    Función principal para análisis básico del dataset BraTS2020
    """
    # Rutas específicas para BraTS2020
    ruta_csv = "./brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv"
    directorio_imagenes = "./brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    
    # Valores que se consideran faltantes
    valores_faltantes = ['', 'NA', 'N/A', 'NULL', '-', '?', 'Missing', 'No data']
    
    try:
        # Verificar que los archivos existen
        if not os.path.exists(ruta_csv):
            print(f"Error: No se encuentra el archivo {ruta_csv}")
            return None
            
        if not os.path.exists(directorio_imagenes):
            print(f"Error: No se encuentra el directorio {directorio_imagenes}")
            return None
        
        # Realizar el análisis
        df_completo = visualizar_valores_faltantes_brats(
            ruta_csv, directorio_imagenes, valores_faltantes
        )
        
        return df_completo
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        return None

def analizar_brats2020_alternativo():
    """
    Función principal para análisis con visualizaciones alternativas
    """
    # Rutas específicas para BraTS2020
    ruta_csv = "./brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv"
    directorio_imagenes = "./brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    
    # Valores que se consideran faltantes
    valores_faltantes = ['', 'NA', 'N/A', 'NULL', '-', '?', 'Missing', 'No data']
    
    try:
        # Verificar que los archivos existen
        if not os.path.exists(ruta_csv):
            print(f"Error: No se encuentra el archivo {ruta_csv}")
            return None
            
        if not os.path.exists(directorio_imagenes):
            print(f"Error: No se encuentra el directorio {directorio_imagenes}")
            return None
        
        # Crear dataset completo
        df_completo, df_original = crear_dataset_completo_brats(
            ruta_csv, directorio_imagenes, valores_faltantes
        )
        
        print("\n" + "="*80)
        print("CREANDO VISUALIZACIONES ALTERNATIVAS DE VALORES FALTANTES")
        print("="*80)
        
        # 1. Gráficos de dona
        print("\n1. Creando gráficos de dona...")
        crear_grafico_dona_valores_faltantes(df_completo)
        
        # 2. Heatmap de completitud
        print("\n2. Creando heatmap de completitud...")
        crear_heatmap_completitud(df_completo)
        
        # 3. Gráfico de área apilada
        print("\n3. Creando gráfico de área apilada...")
        crear_grafico_area_apilada(df_completo)
        
        # 4. Gráfico de radar
        print("\n4. Creando gráfico de radar...")
        crear_grafico_radar_completitud(df_completo)
        
        return df_completo
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        return None

def visualizar_segmentacion_3d():
    """
    Función para visualizar segmentación 3D
    """
    seg_path = "./BCBM-RadioGenomics_Images_Masks_Dec2024/BCBM-RadioGenomics-5-0/BCBM-RadioGenomics-5-0_mask_L-cerebellar-cavity.nii.gz"
    plot_segmentation_3D_with_shadows(seg_path)

# Ejecutar análisis
if __name__ == "__main__":
    # Análisis básico
    df_resultado_basico = analizar_brats2020()
    
    # Análisis con visualizaciones alternativas
    df_resultado_alternativo = analizar_brats2020_alternativo()
    
    # Visualización 3D
    visualizar_segmentacion_3d()
