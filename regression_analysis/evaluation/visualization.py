# -*- coding: utf-8 -*-
"""
Módulo para visualización de resultados de regresión
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import validation_curve, learning_curve
from typing import Tuple, Optional, List

def plot_real_vs_predicted(
    target: str, 
    target_name: str, 
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    min_val_real: Optional[float] = None, 
    max_val_real: Optional[float] = None, 
    noise_std: Optional[float] = None
):
    """
    Visualiza valores reales vs. predichos con opción de desnormalizar.
    
    Args:
        target: nombre de la variable objetivo normalizada
        target_name: nombre de la variable original
        train_df, val_df, test_df: DataFrames con columnas [target, pred_target]
        min_val_real, max_val_real: valores reales para desnormalización
        noise_std: desviación estándar del ruido gaussiano
    """
    # Si no se especifican los valores reales, calcularlos automáticamente
    if min_val_real is None or max_val_real is None:
        min_val_real = train_df[target_name].min()
        max_val_real = train_df[target_name].max()

    # Crear copias con etiquetas
    train_df_ = train_df[[target, f"pred_{target}"]].copy()
    train_df_["set"] = "Train"

    val_df_ = val_df[[target, f"pred_{target}"]].copy()
    val_df_["set"] = "Validation"

    test_df_ = test_df[[target, f"pred_{target}"]].copy()
    test_df_["set"] = "Test"

    # Unir y filtrar
    combined_df = pd.concat([train_df_, val_df_, test_df_])
    mask = combined_df[target].notnull() & combined_df[f"pred_{target}"].notnull()
    filtered_df = combined_df.loc[mask]

    # Desnormalizar usando los valores reales especificados
    scale = max_val_real - min_val_real
    filtered_df = filtered_df.copy()
    filtered_df.loc[:, target] = filtered_df[target].clip(0, 1)
    filtered_df.loc[:, f"pred_{target}"] = filtered_df[f"pred_{target}"].clip(0, 1)
    
    # Desnormalizar
    filtered_df.loc[:, "Real"] = filtered_df[target] * scale + min_val_real
    filtered_df.loc[:, "Predicted"] = filtered_df[f"pred_{target}"] * scale + min_val_real

    # Añadir ruido gaussiano a ambos ejes
    if noise_std is None:
        data_range = max_val_real - min_val_real
        noise_std = data_range * 0.01

    np.random.seed(42)
    noise_x = np.random.normal(0, noise_std, size=len(filtered_df))
    noise_y = np.random.normal(0, noise_std, size=len(filtered_df))
    
    filtered_df["Real_with_noise"] = filtered_df["Real"] + noise_x
    filtered_df["Predicted_with_noise"] = filtered_df["Predicted"] + noise_y

    # Gráfico
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=filtered_df,
        x="Real_with_noise",
        y="Predicted_with_noise",
        hue="set",
        palette={"Train": "blue", "Validation": "orange", "Test": "green"},
        alpha=0.7
    )

    # Línea ideal
    plt.plot([min_val_real, max_val_real], [min_val_real, max_val_real], 
             'r--', label="Ideal (y = x)", linewidth=2)

    plt.xlabel(f"Valor real de {target_name}")
    plt.ylabel(f"Predicción de {target_name}")
    plt.title(f"Real vs. Predicho: {target_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_validation_curve_generic(
    estimator, X, y, param_name: str, param_range: List,
    cv: int = 5, scoring: str = 'neg_mean_absolute_error',
    figsize: Tuple[int, int] = (12, 5), 
    colors: List[str] = ['deepskyblue', 'crimson'],
    title: Optional[str] = None
):
    """
    Función genérica para graficar curvas de validación variando un hiperparámetro.
    """
    # Calcular curvas de validación
    train_scores, validation_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1
    )
    
    # Calcular medias y desviaciones estándar
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)
    
    # Si la métrica es negativa, convertir a positiva
    if scoring.startswith('neg_'):
        train_mean = -train_mean
        validation_mean = -validation_mean
        ylabel = scoring.replace('neg_', '').replace('_', ' ').title()
    else:
        ylabel = scoring.replace('_', ' ').title()
    
    # Crear el gráfico
    plt.figure(figsize=figsize)
    
    # Líneas principales
    plt.plot(param_range, train_mean, 'o-', color=colors[0], 
             label='Training Score', linewidth=2, markersize=6)
    plt.plot(param_range, validation_mean, 'o-', color=colors[1], 
             label='Validation Score', linewidth=2, markersize=6)
    
    # Áreas de incertidumbre
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color=colors[0])
    plt.fill_between(param_range, validation_mean - validation_std, 
                     validation_mean + validation_std, alpha=0.2, color=colors[1])
    
    # Configuración del gráfico
    if title is None:
        title = f'Validation Curve - {param_name}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Si el rango de parámetros es logarítmico, usar escala log
    if len(param_range) > 1 and param_range[1] / param_range[0] > 2:
        plt.xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Encontrar el mejor parámetro
    best_idx = np.argmax(validation_mean)
    best_param = param_range[best_idx]
    best_score = validation_mean[best_idx]
    
    print(f"Mejor {param_name}: {best_param}")
    print(f"Mejor score de validación: {best_score:.4f}")
    
    return train_scores, validation_scores

def plot_learning_curve_generic(
    estimator, X, y, cv: int = 5,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    scoring: str = 'neg_mean_absolute_error',
    figsize: Tuple[int, int] = (10, 4), 
    colors: List[str] = ['deepskyblue', 'crimson'],
    title: Optional[str] = None
):
    """
    Función genérica para graficar curvas de aprendizaje variando el tamaño del dataset.
    """
    # Calcular curvas de aprendizaje
    train_sizes_abs, train_scores, validation_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes, cv=cv, 
        scoring=scoring, n_jobs=-1, random_state=42
    )
    
    # Calcular medias y desviaciones estándar
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)
    
    # Si la métrica es negativa, convertir a positiva
    if scoring.startswith('neg_'):
        train_mean = -train_mean
        validation_mean = -validation_mean
        ylabel = scoring.replace('neg_', '').replace('_', ' ').title()
    else:
        ylabel = scoring.replace('_', ' ').title()
    
    # Crear el gráfico
    plt.figure(figsize=figsize)
    
    # Líneas principales
    plt.plot(train_sizes_abs, train_mean, color='deepskyblue', 
             label='Train', linewidth=1.2, alpha=0.9)
    plt.plot(train_sizes_abs, validation_mean, color='crimson', 
             label='Val', linewidth=1.2, alpha=0.9)
    
    # Configuración del gráfico
    if title is None:
        title = "Loss"
    
    plt.title(f"{title}\nTrain: {train_mean[-1]:.4f} | Val: {validation_mean[-1]:.4f}", 
              fontsize=10, fontweight='normal', pad=10)
    
    plt.xlabel('Training Set Size', fontsize=9)
    plt.ylabel(ylabel, fontsize=9)
    plt.legend(loc='upper right', fontsize=9, frameon=True, 
               fancybox=False, shadow=False, framealpha=1.0)
    plt.xlim(train_sizes_abs[0], train_sizes_abs[-1])
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.15)
    plt.show()
    
    # Mostrar información final
    print(f"Score final de entrenamiento: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
    print(f"Score final de validación: {validation_mean[-1]:.4f} ± {validation_std[-1]:.4f}")
    print(f"Gap de generalización: {abs(train_mean[-1] - validation_mean[-1]):.4f}")
    
    return train_sizes_abs, train_scores, validation_scores
