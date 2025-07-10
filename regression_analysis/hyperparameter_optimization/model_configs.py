# -*- coding: utf-8 -*-
"""
Configuraciones de modelos para búsqueda de hiperparámetros
"""
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from scipy.stats import uniform, randint

def get_models_and_params(config_seed: int = 42) -> dict:
    """
    Obtiene las configuraciones de modelos y sus espacios de hiperparámetros
    """
    return {
        'SVR_linear': {
            'model': make_pipeline(StandardScaler(), SVR(kernel='linear')),
            'params': {
                'svr__C': [0.1, 1.0, 10.0, 100.0],
                'svr__epsilon': [0.001, 0.01, 0.1, 1.0]
            }
        },
        'Ridge': {
            'model': make_pipeline(StandardScaler(), Ridge()),
            'params': {
                'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(
                random_state=config_seed, 
                n_jobs=1,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'params': {
                'n_estimators': [80, 100, 120, 150],
                'max_depth': [15, 18, 20, 22, 25],
                'max_features': ['sqrt', 'log2', 0.7, 0.8, 0.9],
                'min_samples_split': [3, 5, 7],
                'min_samples_leaf': [1, 2, 3]
            }
        },
        'CatBoost': {
            'model': CatBoostRegressor(
                loss_function="MAE",
                random_seed=config_seed,
                verbose=0,
                thread_count=1,
                iterations=100,
                early_stopping_rounds=20,
                bootstrap_type='Bernoulli',
                subsample=0.8
            ),
            'params': {
                'learning_rate': [0.1, 0.2],
                'depth': [3, 4],
                'l2_leaf_reg': [5, 10]
            }
        },
        'SVR_RBF': {
            'model': make_pipeline(StandardScaler(), SVR(kernel='rbf')),
            'params': {
                'svr__C': [1.5, 2.0, 2.5, 3.0],
                'svr__epsilon': [0.005, 0.01, 0.015, 0.02],
                'svr__gamma': [0.0001, 0.00022, 0.0005, 0.001, 0.002]
            }
        },
        'HistGradientBoosting': {
            'model': HistGradientBoostingRegressor(
                random_state=config_seed,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20
            ),
            'params': {
                'max_iter': [250, 300, 350, 400],
                'learning_rate': [0.08, 0.1, 0.12, 0.15],
                'max_depth': [None, 5, 7, 10],
                'min_samples_leaf': [15, 20, 25, 30],
                'l2_regularization': [0.0, 0.01, 0.1]
            }
        },
        'Ridge_Polynomial': {
            'model': make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=2, include_bias=False),
                Ridge()
            ),
            'params': {
                'ridge__alpha': [0.1, 1.0, 10.0, 100.0]
            }
        },
        'MLPRegressor': {
            'model': make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    early_stopping=True,
                    random_state=config_seed,
                    max_iter=500,
                    validation_fraction=0.2,
                    n_iter_no_change=20
                )
            ),
            'params': {
                'mlpregressor__hidden_layer_sizes': [
                    (50,), (100,), (150,),
                    (50, 25), (100, 50), (150, 75)
                ],
                'mlpregressor__alpha': [0.0001, 0.001, 0.01, 0.1],
                'mlpregressor__learning_rate_init': [0.001, 0.01, 0.1]
            }
        }
    }

def get_random_params() -> dict:
    """
    Obtiene las distribuciones para RandomizedSearchCV
    """
    return {
        'SVR_linear': {
            'svr__C': uniform(0.1, 100),
            'svr__epsilon': uniform(0.001, 1)
        },
        'Ridge': {
            'ridge__alpha': uniform(0.01, 100)
        },
        'RandomForest': {
            'n_estimators': randint(80, 150),
            'max_depth': randint(15, 25),
            'max_features': uniform(0.6, 0.4),
            'min_samples_split': randint(3, 8),
            'min_samples_leaf': randint(1, 4)
        },
        'CatBoost': {
            'learning_rate': uniform(0.1, 0.2),
            'depth': randint(3, 5),
            'l2_leaf_reg': uniform(3, 10)
        },
        'SVR_RBF': {
            'svr__C': uniform(1.0, 3.0),
            'svr__epsilon': uniform(0.005, 0.02),
            'svr__gamma': uniform(0.0001, 0.002)
        },
        'HistGradientBoosting': {
            'max_iter': randint(250, 400),
            'learning_rate': uniform(0.05, 0.15),
            'max_depth': randint(3, 12),
            'min_samples_leaf': randint(15, 35),
            'l2_regularization': uniform(0.0, 0.2)
        },
        'Ridge_Polynomial': {
            'ridge__alpha': uniform(0.1, 100)
        },
        'MLPRegressor': {
            'mlpregressor__alpha': uniform(0.0001, 0.1),
            'mlpregressor__learning_rate_init': uniform(0.001, 0.1)
        }
    }
