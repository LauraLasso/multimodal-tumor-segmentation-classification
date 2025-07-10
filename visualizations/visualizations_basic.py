import matplotlib.pyplot as plt
import numpy as np
from missing_values_analysis import calcular_estadisticas_faltantes, generar_resumen_estadistico
from data_utils import crear_dataset_completo_brats

def crear_graficos_barras_faltantes(df_completo):
    """
    Crear gráficos de barras para valores faltantes (visualización original)
    """
    info_columnas = calcular_estadisticas_faltantes(df_completo)
    
    # Filtrar columnas con valores faltantes para visualización
    columnas_con_faltantes = [col for col in info_columnas.keys() if info_columnas[col]['faltantes'] > 0]
    
    if not columnas_con_faltantes:
        print("¡Excelente! No hay valores faltantes en el dataset.")
        return df_completo
    
    # Crear visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Datos para los gráficos
    nombres_columnas = columnas_con_faltantes
    conteos_faltantes = [info_columnas[col]['faltantes'] for col in columnas_con_faltantes]
    porcentajes_faltantes = [info_columnas[col]['porcentaje_faltante'] for col in columnas_con_faltantes]
    
    # Gráfico de conteos absolutos
    bars1 = ax1.bar(range(len(nombres_columnas)), conteos_faltantes, 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Valores Faltantes por Variable (Conteo Absoluto)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Variables', fontsize=12)
    ax1.set_ylabel('Número de Valores Faltantes', fontsize=12)
    ax1.set_xticks(range(len(nombres_columnas)))
    ax1.set_xticklabels(nombres_columnas, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Añadir etiquetas en las barras
    for i, (bar, count) in enumerate(zip(bars1, conteos_faltantes)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(conteos_faltantes)*0.01, 
                str(int(count)), ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Gráfico de porcentajes
    bars2 = ax2.bar(range(len(nombres_columnas)), porcentajes_faltantes, 
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Valores Faltantes por Variable (Porcentaje)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Variables', fontsize=12)
    ax2.set_ylabel('Porcentaje de Valores Faltantes (%)', fontsize=12)
    ax2.set_xticks(range(len(nombres_columnas)))
    ax2.set_xticklabels(nombres_columnas, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Añadir etiquetas en las barras
    for i, (bar, percentage) in enumerate(zip(bars2, porcentajes_faltantes)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(porcentajes_faltantes)*0.01, 
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.show()

def visualizar_valores_faltantes_brats(ruta_csv_survival, directorio_imagenes, valores_faltantes=None):
    """
    Función principal para visualización básica de valores faltantes
    """
    # Crear dataset completo
    df_completo, df_original = crear_dataset_completo_brats(
        ruta_csv_survival, directorio_imagenes, valores_faltantes
    )
    
    # Crear gráficos de barras
    crear_graficos_barras_faltantes(df_completo)
    
    # Generar resumen estadístico
    info_columnas = calcular_estadisticas_faltantes(df_completo)
    generar_resumen_estadistico(df_completo, df_original, info_columnas)
    
    return df_completo
