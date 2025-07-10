# -*- coding: utf-8 -*-
"""
Módulo para evaluación completa de modelos
"""
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Callable, Optional 
from regression_analysis.evaluation.visualization import plot_learning_curve_generic, plot_real_vs_predicted

def evaluate_model_complete(
    estimator, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
    features: List[str], targets: List[Tuple], 
    metric_func: Callable = mean_absolute_error, 
    plot_curves: bool = True, 
    plot_regression: bool = True, 
    cv: int = 5, 
    scoring: str = 'neg_mean_absolute_error',
    use_scaler: bool = True
):
    """
    Función completa que evalúa un modelo con curvas de entrenamiento y gráficos de regresión.
    
    Parameters:
    -----------
    estimator : sklearn estimator
        Modelo de sklearn a evaluar
    train_df, val_df, test_df : pandas.DataFrame
        DataFrames de entrenamiento, validación y test
    features : list
        Lista de nombres de características
    targets : list
        Lista de tuplas con información de targets [(target, c), ...]
    metric_func : function
        Función de métrica para evaluación
    plot_curves : bool
        Si graficar curvas de aprendizaje
    plot_regression : bool
        Si graficar gráficos de regresión real vs predicho
    cv : int
        Número de folds para validación cruzada
    scoring : str
        Métrica para curvas de aprendizaje
    use_scaler : bool
        Si aplicar StandardScaler a las features
    """
    overall_scores = {
        "train": {},
        "val": {},
        "test": {}
    }
    
    for target_info in targets:
        if len(target_info) == 3:
            target, c, w = target_info
        elif len(target_info) == 2:
            target, c = target_info
            w = 0.5
        else:
            raise ValueError(f"Formato de target no válido: {target_info}")
            
        print(f"\n{'='*60}")
        print(f"PROCESANDO TARGET: {target}")
        print(f"{'='*60}")
        
        # Filtrar nulos
        train_target_df = train_df[train_df[target].notnull()]
        val_target_df = val_df[val_df[target].notnull()]
        test_target_df = test_df[test_df[target].notnull()]
        
        # Aplicar StandardScaler si se solicita
        if use_scaler:
            print("Aplicando StandardScaler a las features...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_target_df[features])
            X_val = scaler.transform(val_target_df[features])
            X_test = scaler.transform(test_target_df[features])
            
            X_train_df = pd.DataFrame(X_train, columns=features, index=train_target_df.index)
            X_val_df = pd.DataFrame(X_val, columns=features, index=val_target_df.index)
            X_test_df = pd.DataFrame(X_test, columns=features, index=test_target_df.index)
        else:
            X_train_df = train_target_df[features]
            X_val_df = val_target_df[features]
            X_test_df = test_target_df[features]
        
        # Configurar modelo
        model = clone(estimator)
        if hasattr(model, 'C'):
            model.C = c
        
        # Entrenar modelo
        model.fit(X_train_df, train_target_df[target])
        
        # Predicciones
        train_preds = model.predict(X_train_df)
        val_preds = model.predict(X_val_df)
        test_preds = model.predict(X_test_df)
        
        # Guardar predicciones
        train_df.loc[train_target_df.index, f"pred_{target}"] = train_preds
        val_df.loc[val_target_df.index, f"pred_{target}"] = val_preds
        test_df.loc[test_target_df.index, f"pred_{target}"] = test_preds
        
        # Calcular métricas
        train_score = metric_func(train_target_df[target].values, train_preds)
        val_score = metric_func(val_target_df[target].values, val_preds)
        test_score = metric_func(test_target_df[target].values, test_preds)
        
        overall_scores["train"][target] = train_score
        overall_scores["val"][target] = val_score
        overall_scores["test"][target] = test_score
        
        print(f"\nSCORES FINALES:")
        print(f"Train score: {round(train_score, 4)}")
        print(f"Validation score: {round(val_score, 4)}")
        print(f"Test score: {round(test_score, 4)}")
        
        # Graficar curvas de aprendizaje si se solicita
        if plot_curves:
            print(f"\nGenerando curvas de aprendizaje para {target}...")
            if use_scaler:
                model_for_curves = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', clone(estimator))
                ])
                if hasattr(estimator, 'C'):
                    model_for_curves.set_params(model__C=c)
                
                plot_learning_curve_generic(
                    model_for_curves, 
                    train_target_df[features], 
                    train_target_df[target],
                    cv=cv,
                    scoring=scoring,
                    title=f'Loss'
                )
            else:
                plot_learning_curve_generic(
                    clone(model), 
                    X_train_df, 
                    train_target_df[target],
                    cv=cv,
                    scoring=scoring,
                    title=f'Loss'
                )
        
        # Graficar regresión real vs predicho si se solicita
        if plot_regression:
            print(f"\nGenerando gráfico de regresión para {target}...")
            target_name = target.replace('_normalized', '')
            
            # Filtrar negativos
            train_mask = (train_df[target] >= 0) & (train_df[f"pred_{target}"] >= 0)
            val_mask = (val_df[target] >= 0) & (val_df[f"pred_{target}"] >= 0)
            test_mask = (test_df[target] >= 0) & (test_df[f"pred_{target}"] >= 0)

            train_filtered = train_df[train_mask]
            val_filtered = val_df[val_mask]
            test_filtered = test_df[test_mask]
            
            # Configurar valores para desnormalización
            min_val_real = 5
            max_val_real = 1767
            if "age" in target.lower():
                min_val_real = 18.975
                max_val_real = 86.652

            plot_real_vs_predicted(
                target, target_name, train_filtered, val_filtered, test_filtered, 
                min_val_real=min_val_real, max_val_real=max_val_real, noise_std=None
            )
    
    return overall_scores

def plot_model_comparison_complete(
    models_config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
    features: List[str], targets: List[Tuple],
    metric_func: Callable = mean_absolute_error, 
    cv: int = 5, 
    scoring: str = 'neg_mean_absolute_error',
    figsize: Tuple[int, int] = (15, 6), 
    colors: Optional[List[str]] = None
):
    """
    Función para comparar múltiples modelos con curvas de aprendizaje y gráficos de regresión.
    """
    if colors is None:
        colors = ['deepskyblue', 'crimson', 'green', 'orange', 'purple', 'brown']
    
    # Evaluar cada modelo
    all_scores = {}
    for model_name, estimator in models_config.items():
        print(f"\n{'='*80}")
        print(f"EVALUANDO MODELO: {model_name}")
        print(f"{'='*80}")
        
        scores = evaluate_model_complete(
            estimator, train_df.copy(), val_df.copy(), test_df.copy(), 
            features, targets, metric_func=metric_func, 
            plot_curves=False, plot_regression=True, cv=cv, scoring=scoring
        )
        all_scores[model_name] = scores
    
    # Resumen comparativo
    print(f"\n{'='*80}")
    print("RESUMEN COMPARATIVO DE TODOS LOS MODELOS")
    print(f"{'='*80}")
    
    for target_info in targets:
        if len(target_info) >= 2:
            target = target_info[0]
            print(f"\n--- {target} ---")
            print(f"{'Modelo':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Gap':<10}")
            print("-" * 60)
            
            for model_name in models_config.keys():
                train_score = all_scores[model_name]["train"][target]
                val_score = all_scores[model_name]["val"][target]
                test_score = all_scores[model_name]["test"][target]
                gap = abs(train_score - val_score)
                
                print(f"{model_name:<20} {train_score:<10.4f} {val_score:<10.4f} {test_score:<10.4f} {gap:<10.4f}")
    
    return all_scores

def evaluate_trained_models(
    best_models: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    features: list,
    targets: list
):
    """
    Evaluar todos los modelos entrenados guardados en best_models
    
    Args:
        best_models: Diccionario con estructura {target: {model_name: trained_model}}
        train_df, val_df, test_df: DataFrames con los datos
        features: Lista de nombres de características
        targets: Lista de tuplas con targets [(target_name, c_param), ...]
    """
    print("="*100)
    print("EVALUACIÓN COMPLETA DE MODELOS YA ENTRENADOS")
    print("="*100)
    
    all_evaluation_scores = {}
    
    # Iterar sobre cada target
    for target_info in targets:
        target = target_info[0] if isinstance(target_info, tuple) else target_info
        
        print(f"\n{'='*80}")
        print(f"EVALUANDO MODELOS PARA TARGET: {target}")
        print(f"{'='*80}")
        
        if target not in best_models:
            print(f"No se encontraron modelos para el target: {target}")
            continue
        
        target_scores = {}
        target_models = best_models[target]
        
        # Evaluar cada modelo para este target
        for model_name, trained_model in target_models.items():
            print(f"\n{'-'*60}")
            print(f"EVALUANDO: {model_name} para {target}")
            print(f"{'-'*60}")
            
            try:
                # Usar evaluate_model_complete para el modelo ya entrenado
                scores = evaluate_model_complete(
                    estimator=trained_model,
                    train_df=train_df.copy(),
                    val_df=val_df.copy(),
                    test_df=test_df.copy(),
                    features=features,
                    targets=[target_info],  # Solo evaluar este target
                    metric_func=mean_absolute_error,
                    plot_curves=False,      # Mostrar curvas de aprendizaje
                    plot_regression=True,  # Mostrar gráficos real vs predicho
                    cv=5,
                    scoring='neg_mean_absolute_error',
                    use_scaler=False       # Los modelos ya están entrenados
                )
                
                target_scores[model_name] = scores
                
                print(f"{model_name} evaluado correctamente")
                
            except Exception as e:
                print(f"Error evaluando {model_name}: {e}")
                continue
        
        all_evaluation_scores[target] = target_scores
    
    return all_evaluation_scores