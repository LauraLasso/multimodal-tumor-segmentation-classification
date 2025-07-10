import pandas as pd
import numpy as np
from data_utils import crear_dataset_completo_brats

def calcular_estadisticas_faltantes(df_completo):
    """
    Calcular estadísticas detalladas de valores faltantes
    """
    info_columnas = {}
    for col in df_completo.columns:
        if col != 'Brats20ID':  # Excluir la columna ID
            total_pacientes = len(df_completo)
            valores_presentes = df_completo[col].notna().sum()
            valores_faltantes = df_completo[col].isna().sum()
            
            info_columnas[col] = {
                'total': total_pacientes,
                'presentes': valores_presentes,
                'faltantes': valores_faltantes,
                'porcentaje_faltante': (valores_faltantes / total_pacientes) * 100
            }
    
    return info_columnas

def generar_resumen_estadistico(df_completo, df_original, info_columnas):
    """
    Generar resumen estadístico detallado
    """
    print("\n" + "="*80)
    print("RESUMEN DETALLADO DE VALORES FALTANTES - DATASET BraTS2020")
    print("="*80)
    print(f"Total de pacientes en el directorio: {len(df_completo)}")
    print(f"Pacientes con datos en survival_info.csv: {len(df_original)}")
    print(f"Pacientes SIN datos de supervivencia: {len(df_completo) - len(df_original)}")
    print(f"Columnas analizadas: {len(info_columnas)}")
    
    print("\nDetalle por columna:")
    print("-" * 80)
    print(f"{'Variable':<25} {'Completos':<10} {'Faltantes':<10} {'% Faltante':<12} {'Estado'}")
    print("-" * 80)
    
    for col in ['Age', 'Survival_days', 'Extent_of_Resection']:  # Orden específico para BraTS
        if col in info_columnas:
            info = info_columnas[col]
            status = "✓ COMPLETA" if info['faltantes'] == 0 else "⚠ INCOMPLETA"
            print(f"{col:<25} {info['presentes']:<10} {info['faltantes']:<10} {info['porcentaje_faltante']:<11.1f}% {status}")
    
    # Análisis adicional específico para BraTS
    print(f"\n" + "="*80)
    print("ANÁLISIS ESPECÍFICO BRATS2020")
    print("="*80)
    
    # Pacientes con datos completos de supervivencia
    pacientes_datos_completos = df_completo.dropna(subset=['Age', 'Survival_days']).shape[0]
    print(f"Pacientes con datos completos de supervivencia (Age + Survival_days): {pacientes_datos_completos}")
    
    # Pacientes solo con imágenes
    pacientes_solo_imagenes = len(df_completo) - len(df_original)
    print(f"Pacientes que solo tienen imágenes (sin datos clínicos): {pacientes_solo_imagenes}")
    
    if 'Extent_of_Resection' in df_completo.columns:
        pacientes_con_reseccion = df_completo['Extent_of_Resection'].notna().sum()
        print(f"Pacientes con información de extensión de resección: {pacientes_con_reseccion}")
