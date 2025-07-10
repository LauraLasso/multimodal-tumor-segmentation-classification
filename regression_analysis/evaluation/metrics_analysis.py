# -*- coding: utf-8 -*-
"""
Módulo para análisis de métricas y creación de tablas
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error
from typing import Dict, List, Tuple, Any

def get_metric_info_from_results(results: Dict) -> tuple:
    """
    Detecta qué métrica se usó basándose en las claves de detailed_fits_table
    """
    # Buscar en los resultados las claves que indican la métrica
    for target_results in results.values():
        for model_results in target_results.values():
            if 'cv_results' in model_results:
                # Intentar detectar por el nombre de la métrica
                return 'MAE', mean_absolute_error
    
    # Fallback: detectar por columnas si se pasa detailed_fits_table
    return 'MAE', mean_absolute_error  # Default

def detect_metric_from_detailed_fits(detailed_fits_table: List[Dict]) -> tuple:
    """
    Detecta la métrica utilizada basándose en los nombres de columnas
    """
    if not detailed_fits_table:
        return 'MAE', mean_absolute_error
    
    # Revisar las claves del primer elemento
    first_row = detailed_fits_table[0]
    
    if any('MSE' in key for key in first_row.keys()):
        from sklearn.metrics import mean_squared_error
        return 'MSE', mean_squared_error
    else:
        return 'MAE', mean_absolute_error


def create_metrics_table(
    best_models: Dict, 
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: List[str], 
    targets: List[Tuple], 
    results: Dict,
    metric_func=mean_absolute_error  # Nuevo parámetro
) -> pd.DataFrame:
    """
    Crea una tabla con métricas de train, val y test para todos los modelos
    """
    # Detectar nombre de la métrica
    if metric_func == mean_absolute_error:
        metric_name = 'MAE'
    else:
        # Importar MSE para comparación
        from sklearn.metrics import mean_squared_error
        if metric_func == mean_squared_error:
            metric_name = 'MSE'
        else:
            metric_name = 'Error'
    
    all_results = []
    
    for target_info in targets:
        target = target_info[0] if isinstance(target_info, tuple) else target_info
        
        if target not in best_models:
            continue
            
        print(f"\nProcesando métricas para {target}...")
        
        # Preparar datos limpios
        train_clean = train_df[[target] + features].dropna()
        val_clean = val_df[[target] + features].dropna()
        test_clean = test_df[[target] + features].dropna()
        
        X_train = train_clean[features]
        y_train = train_clean[target]
        X_val = val_clean[features]
        y_val = val_clean[target]
        X_test = test_clean[features]
        y_test = test_clean[target]
        
        target_models = best_models[target]
        
        # Evaluar cada modelo
        for model_name, model in target_models.items():
            try:
                # Hacer predicciones
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)
                test_preds = model.predict(X_test)
                
                # Calcular métricas usando la función pasada
                train_metric = metric_func(y_train, train_preds)
                val_metric = metric_func(y_val, val_preds)
                test_metric = metric_func(y_test, test_preds)
                
                # Obtener hiperparámetros
                best_params = results[target][model_name]['best_params']
                cv_metric = results[target][model_name]['best_score']
                
                # Formatear hiperparámetros
                if best_params:
                    params_str = ', '.join([
                        f"{k.replace('svr__', '').replace('ridge__', '').replace('mlpregressor__', '')}: {v}" 
                        for k, v in best_params.items()
                    ])
                else:
                    params_str = "Parámetros por defecto"
                
                # Agregar a resultados con nombres dinámicos
                all_results.append({
                    'Target': target.replace('_normalized', ''),
                    'Modelo': model_name,
                    'Hiperparámetros': params_str,
                    f'CV {metric_name}': f"{cv_metric:.4f}",
                    f'Train {metric_name}': f"{train_metric:.4f}",
                    f'Val {metric_name}': f"{val_metric:.4f}",
                    f'Test {metric_name}': f"{test_metric:.4f}",
                    'Train Samples': len(X_train),
                    'Val Samples': len(X_val),
                    'Test Samples': len(X_test)
                })
                
            except Exception as e:
                print(f"Error evaluando {model_name} para {target}: {e}")
                continue
    
    # Crear DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Mostrar tabla completa
    print("\n" + "="*150)
    print("TABLA COMPLETA DE MÉTRICAS - TODOS LOS MODELOS")
    print("="*150)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    print(results_df.to_string(index=False))
    
    # Mostrar tabla separada por target
    for target_info in targets:
        target = target_info[0] if isinstance(target_info, tuple) else target_info
        target_display = target.replace('_normalized', '')
        target_data = results_df[results_df['Target'] == target_display]
        
        if not target_data.empty:
            print(f"\n" + "="*120)
            print(f"MÉTRICAS PARA {target_display.upper()}")
            print("="*120)
            
            # Ordenar por Val metric
            val_col = f'Val {metric_name}'
            target_data_sorted = target_data.sort_values(val_col).reset_index(drop=True)
            target_data_sorted.index = target_data_sorted.index + 1
            
            # Mostrar tabla formateada con nombres dinámicos
            print(f"{'Rank':<4} {'Modelo':<20} {f'CV {metric_name}':<8} {f'Train {metric_name}':<10} {f'Val {metric_name}':<8} {f'Test {metric_name}':<9} {'Hiperparámetros':<60}")
            print("-" * 120)
            
            for idx, row in target_data_sorted.iterrows():
                print(f"{idx:<4} {row['Modelo']:<20} {row[f'CV {metric_name}']:<8} {row[f'Train {metric_name}']:<10} {row[f'Val {metric_name}']:<8} {row[f'Test {metric_name}']:<9} {row['Hiperparámetros']:<60}")
    
    return results_df


def display_detailed_fits_table(detailed_fits_table: List[Dict], filename: str = 'detailed_fits_table.csv') -> pd.DataFrame:
    """
    Muestra la tabla detallada de todos los fits realizados
    """
    detailed_df = pd.DataFrame(detailed_fits_table)
    
    if detailed_df.empty:
        print("No hay datos de fits para mostrar")
        return detailed_df
    
    # Detectar métrica utilizada
    metric_name, _ = detect_metric_from_detailed_fits(detailed_fits_table)
    
    print("\n" + "="*150)
    print("TABLA DETALLADA DE TODOS LOS FITS REALIZADOS")
    print("="*150)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 60)
    
    # Mostrar tabla por modelo y target
    for target in detailed_df['Target'].unique():
        print(f"\n{'='*120}")
        print(f"TARGET: {target.upper()}")
        print("="*120)
        
        target_data = detailed_df[detailed_df['Target'] == target]
        
        for modelo in target_data['Modelo'].unique():
            modelo_data = target_data[target_data['Modelo'] == modelo].copy()
            
            # Ordenar por CV_Val_métrica (dinámico)
            cv_val_col = f'CV_Val_{metric_name}'
            val_col = f'Val_{metric_name}'
            test_col = f'Test_{metric_name}'
            cv_train_col = f'CV_Train_{metric_name}'
            
            modelo_data[f'{cv_val_col}_float'] = modelo_data[cv_val_col].astype(float)
            modelo_data = modelo_data.sort_values(f'{cv_val_col}_float').reset_index(drop=True)
            
            print(f"\n{modelo}:")
            print("-" * 120)
            print(f"{'Rank':<4} {cv_train_col:<12} {cv_val_col:<10} {val_col:<8} {test_col:<9} {'Parámetros':<70}")
            print("-" * 120)
            
            for idx, row in modelo_data.head(10).iterrows():
                rank = idx + 1
                print(f"{rank:<4} {row[cv_train_col]:<12} {row[cv_val_col]:<10} {row[val_col]:<8} {row[test_col]:<9} {row['Parámetros']:<70}")
    
    # Guardar tabla como CSV
    detailed_df.to_csv(filename, index=False)
    print(f"\n✓ Tabla detallada guardada como '{filename}'")
    
    return detailed_df


def create_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una tabla de comparación entre diferentes targets
    """
    print(f"\n" + "="*100)
    print("COMPARACIÓN ENTRE TARGETS")
    print("="*100)
    
    # Detectar métrica de las columnas del DataFrame
    metric_name = 'MAE'  # Default
    for col in results_df.columns:
        if 'MSE' in col:
            metric_name = 'MSE'
            break
        elif 'MAE' in col:
            metric_name = 'MAE'
            break
    
    comparison_data = []
    models = results_df['Modelo'].unique()
    targets = results_df['Target'].unique()
    
    for model in models:
        model_data = results_df[results_df['Modelo'] == model]
        
        row_data = {'Modelo': model}
        
        for target in targets:
            target_data = model_data[model_data['Target'] == target]
            if not target_data.empty:
                row_data[f'{target} - Val {metric_name}'] = target_data[f'Val {metric_name}'].iloc[0]
                row_data[f'{target} - Test {metric_name}'] = target_data[f'Test {metric_name}'].iloc[0]
        
        if len(row_data) > 1:  # Solo agregar si tiene datos
            comparison_data.append(row_data)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        return comparison_df
    
    return pd.DataFrame()

