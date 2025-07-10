# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import re

def plot_predictions_analysis(
    base_path: str,
    target: str = "gt_sd",
    pred_target: str = "pred_sd",
    analysis_type: str = "survival",  # "survival" o "age"
    figsize: Tuple[int, int] = (6, 6),
    colors: Dict[str, str] = None,
    alpha: float = 0.7,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> pd.DataFrame:
    """
    Crear gráfico de dispersión comparando valores reales vs predichos para train/val/test
    Adaptado para análisis de supervivencia y edad
    
    Args:
        base_path: Ruta base donde están los archivos CSV
        target: Nombre de la columna con valores reales
        pred_target: Nombre de la columna con predicciones
        analysis_type: Tipo de análisis ("survival" o "age")
        figsize: Tamaño de la figura (ancho, alto)
        colors: Diccionario con colores personalizados para cada set
        alpha: Transparencia de los puntos (0-1)
        title: Título del gráfico (se genera automáticamente si es None)
        xlabel: Etiqueta del eje X (se genera automáticamente si es None)
        ylabel: Etiqueta del eje Y (se genera automáticamente si es None)
        save_path: Ruta para guardar el gráfico (opcional)
        show_plot: Si mostrar el gráfico o no
    
    Returns:
        DataFrame combinado con todos los datos filtrados
    """
    
    # Configuración automática según el tipo de análisis
    if analysis_type == "survival":
        default_title = "Real vs. Predicho: Supervivencia"
        default_xlabel = "Valor real de días de supervivencia"
        default_ylabel = "Predicción de días de supervivencia"
    elif analysis_type == "age":
        default_title = "Real vs. Predicho: Edad"
        default_xlabel = "Valor real de edad"
        default_ylabel = "Predicción de edad"
    else:
        # Configuración genérica
        default_title = "Real vs. Predicho"
        default_xlabel = f"Valor real de {target}"
        default_ylabel = f"Predicción de {pred_target}"
    
    # Usar valores por defecto si no se especifican
    title = title or default_title
    xlabel = xlabel or default_xlabel
    ylabel = ylabel or default_ylabel
    
    # Colores por defecto
    if colors is None:
        colors = {"Train": "blue", "Validation": "orange", "Test": "green"}
    
    # Convertir a Path para manejo más robusto
    base_path = Path(base_path)
    
    # Rutas de los archivos CSV
    train_path = base_path / "predictions_train.csv"
    val_path = base_path / "predictions_val.csv"
    test_path = base_path / "predictions_test.csv"
    
    # Verificar que los archivos existen
    for path, name in [(train_path, "train"), (val_path, "validation"), (test_path, "test")]:
        if not path.exists():
            raise FileNotFoundError(f"Archivo {name} no encontrado: {path}")
    
    try:
        # Leer los CSV
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # Verificar que las columnas existen
        for df, name in [(train_df, "train"), (val_df, "validation"), (test_df, "test")]:
            if target not in df.columns:
                raise ValueError(f"Columna '{target}' no encontrada en {name}")
            if pred_target not in df.columns:
                raise ValueError(f"Columna '{pred_target}' no encontrada en {name}")
        
        # Añadir columna 'set' a cada DataFrame
        train_df_ = train_df[[target, pred_target]].copy()
        train_df_["set"] = "Train"
        
        val_df_ = val_df[[target, pred_target]].copy()
        val_df_["set"] = "Validation"
        
        test_df_ = test_df[[target, pred_target]].copy()
        test_df_["set"] = "Test"
        
        # Unir todo
        combined_df = pd.concat([train_df_, val_df_, test_df_], ignore_index=True)
        
        # Filtrar filas válidas
        mask = combined_df[target].notnull() & combined_df[pred_target].notnull()
        filtered_df = combined_df.loc[mask].copy()
        
        if filtered_df.empty:
            raise ValueError("No hay datos válidos después del filtrado")
        
        # Crear gráfico
        plt.figure(figsize=figsize)
        
        # Scatter plot
        sns.scatterplot(
            data=filtered_df,
            x=target,
            y=pred_target,
            hue="set",
            palette=colors,
            alpha=alpha
        )
        
        # Línea ideal (y = x)
        min_val = filtered_df[target].min()
        max_val = filtered_df[target].max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y = x)")
        
        # Configurar gráfico
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Guardar si se especifica ruta
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        
        # Mostrar gráfico
        if show_plot:
            plt.show()
        
        return filtered_df
        
    except Exception as e:
        print(f"Error procesando los datos: {e}")
        raise

def extract_all_samples(log_path, target_epochs, total_samples=295, batch_size=2):
    results = {epoch: {"preds": [], "gt": []} for epoch in target_epochs}
    current_epoch = None
    collecting = False
    
    with open(log_path, "r") as file:
        for line in file:
            # Detectar epoch (mejoramos el patrón para capturar tanto [Train] como [Val])
            epoch_match = re.search(r"Epoch:\s+\[(\d+)\](?:\[|\s)", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                collecting = current_epoch in target_epochs and ("[Val]" in line) #or "[Val]" not in line)
            
            if collecting:
                # Extraer valores predichos (patrón mejorado)
                # if "Predicted value:" in line or "Predicted value" in line:
                #     try:
                #         pred_part = line.split("Predicted value")[1].split("GT value")[0]
                #         preds = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", pred_part)]
                #         results[current_epoch]["preds"].extend(preds)
                #     except Exception as e:
                #         print(f"Error procesando predicciones en línea: {line[:100]}...")
                #         print(f"Error: {str(e)}")
                if "Predicted value:" in line or "Predicted value" in line:
                    try:
                        # Extraer de la línea actual
                        pred_part = line.split("Predicted value")[1].split("GT value")[0]
                        preds = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", pred_part)]

                        # Leer la línea siguiente
                        next_line = next(file, "")
                        preds_extra = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", next_line)]
                        
                        # Unir ambas listas
                        preds.extend(preds_extra)

                        # Guardar resultados
                        results[current_epoch]["preds"].extend(preds)
                    except Exception as e:
                        print(f"Error procesando predicciones en línea: {line[:100]}...")
                        print(f"Error: {str(e)}")

                
                # Extraer valores reales (patrón mejorado)
                # if "GT value:" in line or "GT value" in line:
                #     try:
                #         gt_part = line.split("GT value")[1]
                #         gts = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", gt_part)]
                #         results[current_epoch]["gt"].extend(gts)
                #     except Exception as e:
                #         print(f"Error procesando valores reales en línea: {line[:100]}...")
                #         print(f"Error: {str(e)}")
                elif "GT value:" in line or "GT value" in line:
                    try:
                        # Extraer de la línea actual
                        gt_part = line.split("GT value")[1]
                        gts = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", gt_part)]

                        # Leer la línea siguiente
                        next_line = next(file, "")
                        gts_extra = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", next_line)]

                        # Unir ambas listas
                        gts.extend(gts_extra)

                        # Guardar resultados
                        results[current_epoch]["gt"].extend(gts)
                    except Exception as e:
                        print(f"Error procesando valores reales en línea: {line[:100]}...")
                        print(f"Error: {str(e)}")


    # Verificar y ajustar el número de muestras
    for epoch in target_epochs:
        n_preds = len(results[epoch]["preds"])
        n_gts = len(results[epoch]["gt"])
        
        # Asegurarnos de que tenemos el mismo número de predicciones y valores reales
        min_samples = min(n_preds, n_gts)
        results[epoch]["preds"] = results[epoch]["preds"][:min_samples]
        results[epoch]["gt"] = results[epoch]["gt"][:min_samples]
        
        print(f"Epoch {epoch}: {min_samples} muestras recolectadas")
        if min_samples > total_samples:
            print(f"  ¡Nota: Más muestras de las esperadas, recortando a {total_samples}")
            results[epoch]["preds"] = results[epoch]["preds"][:total_samples]
            results[epoch]["gt"] = results[epoch]["gt"][:total_samples]

    return results

def get_prediction_stats(filtered_df: pd.DataFrame, target: str, pred_target: str) -> Dict:
    """
    Calcular estadísticas de las predicciones por conjunto
    
    Args:
        filtered_df: DataFrame con los datos filtrados
        target: Nombre de la columna con valores reales
        pred_target: Nombre de la columna con predicciones
    
    Returns:
        Diccionario con estadísticas por conjunto
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    
    stats = {}
    
    for set_name in filtered_df['set'].unique():
        set_data = filtered_df[filtered_df['set'] == set_name]
        y_true = set_data[target]
        y_pred = set_data[pred_target]
        
        stats[set_name] = {
            'count': len(set_data),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mean_true': y_true.mean(),
            'mean_pred': y_pred.mean(),
            'std_true': y_true.std(),
            'std_pred': y_pred.std()
        }
    
    return stats

# Función de ejemplo de uso
# def example_usage():
#     """
#     Ejemplo de cómo usar la función
#     """
#     # Uso básico
#     base_path = "Processed_BraTS2020/LViT/Test_session_04.30_04h57"
    
#     # Llamar la función
#     filtered_data = plot_predictions_analysis(
#         base_path=base_path,
#         target="gt_sd",
#         pred_target="pred_sd",
#         title="Real vs. Predicho: Supervivencia",
#         save_path="survival_predictions.png"
#     )
    
#     # Obtener estadísticas
#     stats = get_prediction_stats(filtered_data, "gt_sd", "pred_sd")
    
#     # Mostrar estadísticas
#     for set_name, metrics in stats.items():
#         print(f"\n{set_name}:")
#         print(f"  Muestras: {metrics['count']}")
#         print(f"  RMSE: {metrics['rmse']:.2f}")
#         print(f"  MAE: {metrics['mae']:.2f}")
#         print(f"  R²: {metrics['r2']:.3f}")

# if __name__ == "__main__":
#     example_usage()
