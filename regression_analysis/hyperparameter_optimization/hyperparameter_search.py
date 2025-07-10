# -*- coding: utf-8 -*-
"""
Módulo para búsqueda de hiperparámetros
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from copy import deepcopy
from typing import Dict, List, Tuple, Any
from scipy.stats import uniform, randint
 
from regression_analysis.hyperparameter_optimization.model_configs import get_models_and_params, get_random_params

def hyperparameter_search_cv_improved(
    train_df: pd.DataFrame, 
    features: List[str], 
    targets: List[Tuple], 
    cv: int = 3, 
    search_type: str = 'random',
    max_samples: int = 1000,
    config_seed: int = 42,
    loss=mean_absolute_error
):
    """
    Búsqueda de hiperparámetros mejorada con configuraciones conocidas como buenas
    
    Args:
        train_df: DataFrame de entrenamiento
        features: Lista de características
        targets: Lista de targets
        cv: Número de folds para validación cruzada
        search_type: Tipo de búsqueda ('grid' o 'random')
        max_samples: Máximo número de muestras para usar
        config_seed: Semilla para reproducibilidad
    
    Returns:
        Tuple con (best_models, results, detailed_fits_table)
    """
    # Comprobar la función usada para el entrenamiento, para así actualizar lo que se muestra
    metric_name = get_metric_name(loss)
    # Scorer personalizado para MAE
    mae_scorer = make_scorer(loss, greater_is_better=False)
    
    # Obtener configuraciones de modelos
    models_and_params = get_models_and_params(config_seed)
    random_params = get_random_params()
    
    best_models = {}
    results = {}
    detailed_fits_table = []
    
    # Preparar todos los targets para validación
    all_targets = []
    for target_info in targets:
        target = target_info[0] if isinstance(target_info, tuple) else target_info
        all_targets.append(target)
    
    # Preparar datos de validación y test
    val_clean = train_df[features + all_targets].dropna()  # Usar train_df como referencia
    test_clean = train_df[features + all_targets].dropna()  # Ajustar según necesidad
    
    # Iterar sobre cada target
    for target_info in targets:
        target = target_info[0] if isinstance(target_info, tuple) else target_info
        print(f"\n{'='*60}")
        print(f"BÚSQUEDA MEJORADA PARA: {target}")
        print(f"{'='*60}")
        
        # Preparar datos limpios
        train_clean = train_df[[target] + features].dropna()
        
        print(f"Datos originales: {len(train_clean)}")
        print(f"Target range: [{train_clean[target].min():.4f}, {train_clean[target].max():.4f}]")
        print(f"Target std: {train_clean[target].std():.4f}")
        
        # Muestrear si es necesario
        if len(train_clean) > max_samples:
            train_clean = train_clean.sample(n=max_samples, random_state=config_seed)
            print(f"Datos muestreados a {max_samples} muestras")
        
        X_train = train_clean[features]
        y_train = train_clean[target]
        
        # Preparar datos de val y test para este target
        X_val = val_clean[features]
        y_val = val_clean[target]
        X_test = test_clean[features]
        y_test = test_clean[target]
        
        print(f"Features shape: {X_train.shape}")
        print(f"Features with NaN: {X_train.isnull().sum().sum()}")
        
        target_results = {}
        target_best_models = {}
        
        # Iterar sobre cada modelo
        for model_name, model_config in models_and_params.items():
            print(f"\n--- Optimizando {model_name} (configuración mejorada) ---")
            
            try:
                model = model_config['model']
                
                if search_type == 'grid':
                    search = GridSearchCV(
                        estimator=model,
                        param_grid=model_config['params'],
                        cv=cv,
                        scoring=mae_scorer,
                        n_jobs=1,
                        verbose=1,
                        error_score='raise',
                        return_train_score=True
                    )
                else:  # random search
                    # Más iteraciones para modelos mejorados
                    if model_name in ['RandomForest', 'SVR_RBF', 'HistGradientBoosting']:
                        n_iter = 20
                    else:
                        n_iter = 15
                    
                    if model_name in random_params:
                        search = RandomizedSearchCV(
                            estimator=model,
                            param_distributions=random_params[model_name],
                            n_iter=n_iter,
                            cv=cv,
                            scoring=mae_scorer,
                            n_jobs=1,
                            random_state=config_seed,
                            verbose=1,
                            error_score='raise',
                            return_train_score=True
                        )
                    else:
                        search = GridSearchCV(
                            estimator=model,
                            param_grid=model_config['params'],
                            cv=cv,
                            scoring=mae_scorer,
                            n_jobs=1,
                            verbose=1,
                            error_score='raise',
                            return_train_score=True
                        )
                
                # Ejecutar búsqueda
                search.fit(X_train, y_train)
                
                # Extraer información detallada de todos los fits
                cv_results = search.cv_results_
                
                for i in range(len(cv_results['params'])):
                    params = cv_results['params'][i]
                    params_str = ', '.join([f"{k}: {v}" for k, v in params.items()])
                    
                    cv_train_score = -cv_results['mean_train_score'][i]
                    cv_val_score = -cv_results['mean_test_score'][i]
                    cv_train_std = cv_results['std_train_score'][i]
                    cv_val_std = cv_results['std_test_score'][i]
                    
                    # Entrenar modelo con estos parámetros para métricas en val y test
                    temp_model = deepcopy(model_config['model'])
                    temp_model.set_params(**params)
                    temp_model.fit(X_train, y_train)
                    
                    val_pred = temp_model.predict(X_val)
                    test_pred = temp_model.predict(X_test)
                    
                    val_mae = loss(y_val, val_pred)
                    test_mae = loss(y_test, test_pred)
                    
                    # Agregar a la tabla detallada
                    detailed_fits_table.append({
                        'Target': target,
                        'Modelo': model_name,
                        'Fit_ID': f"{model_name}_{target}_{i+1}",
                        'Parámetros': params_str,
                        f'CV_Train_{metric_name}': f"{cv_train_score:.4f}",
                        'CV_Train_Std': f"{cv_train_std:.4f}",
                        f'CV_Val_{metric_name}': f"{cv_val_score:.4f}",
                        'CV_Val_Std': f"{cv_val_std:.4f}",
                        f'Val_{metric_name}': f"{val_mae:.4f}",
                        f'Test_{metric_name}': f"{test_mae:.4f}",
                        'Rank': i + 1
                    })
                
                # Guardar resultados
                target_results[model_name] = {
                    'best_score': -search.best_score_,
                    'best_params': search.best_params_,
                    'cv_results': search.cv_results_
                }
                
                target_best_models[model_name] = search.best_estimator_
                
                print(f"✓ {model_name} - Mejor {metric_name}: {-search.best_score_:.4f}")
                print(f"  Mejores parámetros: {search.best_params_}")
                print(f"  Total de fits evaluados: {len(cv_results['params'])}")
                
                # Validación adicional
                train_pred = search.best_estimator_.predict(X_train)
                train_mae = mean_absolute_error(y_train, train_pred)
                print(f"  {metric_name} en entrenamiento: {train_mae:.4f}")
                
            except Exception as e:
                print(f"✗ Error con {model_name}: {e}")
                continue
        
        results[target] = target_results
        best_models[target] = target_best_models
    
    return best_models, results, detailed_fits_table

def get_metric_name(loss_func):
    """Detecta el nombre de la métrica basándose en la función de pérdida"""
    if hasattr(loss_func, '__name__'):
        if 'absolute' in loss_func.__name__ or 'mae' in loss_func.__name__.lower():
            return 'MAE'
        elif 'squared' in loss_func.__name__ or 'mse' in loss_func.__name__.lower():
            return 'MSE'
    return 'Error'  # Fallback genérico
