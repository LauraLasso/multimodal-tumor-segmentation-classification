import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def crear_grafico_dona_valores_faltantes(df_completo):
    """
    Crear gráficos de dona para mostrar valores faltantes vs presentes por variable
    """
    # Calcular información por columna (excluyendo ID)
    columnas_analizar = [col for col in df_completo.columns if col != 'Brats20ID']
    
    fig, axes = plt.subplots(1, len(columnas_analizar), figsize=(18, 6))
    if len(columnas_analizar) == 1:
        axes = [axes]
    
    colores_presentes = ['#2ecc71', '#3498db', '#9b59b6']  # Verde, azul, morado
    colores_faltantes = ['#e74c3c', '#e67e22', '#f39c12']  # Rojo, naranja, amarillo
    
    for i, col in enumerate(columnas_analizar):
        valores_presentes = df_completo[col].notna().sum()
        valores_faltantes = df_completo[col].isna().sum()
        total = len(df_completo)
        
        # Datos para el gráfico de dona
        sizes = [valores_presentes, valores_faltantes]
        labels = ['Presentes', 'Faltantes']
        colors = [colores_presentes[i], colores_faltantes[i]]
        
        # Crear gráfico de dona con texto personalizado
        wedges, texts, autotexts = axes[i].pie(sizes, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90,
                                              wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
                                              pctdistance=0.85)
        
        # Personalizar texto de porcentajes
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
            autotext.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))
        
        # Personalizar texto de etiquetas
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        
        # Título y estadísticas en el centro
        axes[i].set_title(f'{col}', fontsize=14, fontweight='bold', pad=20)
        
        # Añadir texto en el centro con mejor contraste
        centre_text = f'{valores_presentes}/{total}\npacientes'
        axes[i].text(0, 0, centre_text, ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.suptitle('Distribución de Valores Presentes vs Faltantes por Variable', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def crear_heatmap_completitud(df_completo):
    """
    Crear un heatmap mostrando la completitud de datos por variable
    """
    # Preparar datos para el heatmap
    columnas_analizar = [col for col in df_completo.columns if col != 'Brats20ID']
    
    # Calcular porcentajes de completitud y faltantes
    completitud_data = []
    for col in columnas_analizar:
        valores_presentes = df_completo[col].notna().sum()
        valores_faltantes = df_completo[col].isna().sum()
        total = len(df_completo)
        
        porcentaje_presente = (valores_presentes / total) * 100
        porcentaje_faltante = (valores_faltantes / total) * 100
        
        completitud_data.append([porcentaje_presente, porcentaje_faltante])
    
    # Crear DataFrame para el heatmap
    heatmap_df = pd.DataFrame(completitud_data, 
                             columns=['Datos Presentes (%)', 'Datos Faltantes (%)'],
                             index=columnas_analizar)
    
    # Crear el heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Usar una paleta de colores personalizada
    cmap = sns.diverging_palette(10, 150, as_cmap=True)
    
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap=cmap, center=50,
                square=True, cbar_kws={"shrink": .8}, ax=ax,
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_title('Mapa de Calor: Completitud de Datos por Variable', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Tipo de Dato', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variables', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def crear_grafico_area_apilada(df_completo):
    """
    Crear gráfico de área apilada mostrando la distribución de completitud
    """
    # Preparar datos
    columnas_analizar = [col for col in df_completo.columns if col != 'Brats20ID']
    
    datos_presentes = []
    datos_faltantes = []
    
    for col in columnas_analizar:
        valores_presentes = df_completo[col].notna().sum()
        valores_faltantes = df_completo[col].isna().sum()
        
        datos_presentes.append(valores_presentes)
        datos_faltantes.append(valores_faltantes)
    
    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(columnas_analizar))
    
    # Crear áreas apiladas
    ax.fill_between(x, 0, datos_presentes, alpha=0.8, color='#2ecc71', label='Datos Presentes')
    ax.fill_between(x, datos_presentes, 
                   [p + f for p, f in zip(datos_presentes, datos_faltantes)], 
                   alpha=0.8, color='#e74c3c', label='Datos Faltantes')
    
    # Personalizar el gráfico
    ax.set_title('Distribución de Completitud de Datos por Variable (Área Apilada)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Variables', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Pacientes', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(columnas_analizar, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Añadir valores en el gráfico
    for i, (presente, faltante) in enumerate(zip(datos_presentes, datos_faltantes)):
        total = presente + faltante
        # Etiqueta para datos presentes
        ax.text(i, presente/2, f'{presente}', ha='center', va='center', 
                fontweight='bold', fontsize=11, color='white')
        # Etiqueta para datos faltantes
        if faltante > 0:
            ax.text(i, presente + faltante/2, f'{faltante}', ha='center', va='center', 
                    fontweight='bold', fontsize=11, color='white')
    
    plt.tight_layout()
    plt.show()

def crear_grafico_radar_completitud(df_completo):
    """
    Crear gráfico de radar mostrando el porcentaje de completitud por variable
    """
    # Preparar datos
    columnas_analizar = [col for col in df_completo.columns if col != 'Brats20ID']
    
    porcentajes_completitud = []
    for col in columnas_analizar:
        valores_presentes = df_completo[col].notna().sum()
        total = len(df_completo)
        porcentaje = (valores_presentes / total) * 100
        porcentajes_completitud.append(porcentaje)
    
    # Configurar el gráfico radar
    angles = np.linspace(0, 2 * np.pi, len(columnas_analizar), endpoint=False).tolist()
    porcentajes_completitud += porcentajes_completitud[:1]  # Cerrar el círculo
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Crear el gráfico
    ax.plot(angles, porcentajes_completitud, 'o-', linewidth=3, color='#3498db', markersize=8)
    ax.fill(angles, porcentajes_completitud, alpha=0.25, color='#3498db')
    
    # Personalizar
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(columnas_analizar, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
    ax.grid(True)
    
    # Añadir valores en cada punto
    for angle, porcentaje, col in zip(angles[:-1], porcentajes_completitud[:-1], columnas_analizar):
        ax.text(angle, porcentaje + 5, f'{porcentaje:.1f}%', 
                ha='center', va='center', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_title('Radar de Completitud de Datos por Variable', 
                 fontsize=16, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.show()
