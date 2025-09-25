"""
Módulo para entrenamiento de modelos.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def calculate_metrics(y_true, y_pred):
    """
    Calcula múltiples métricas de regresión.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

def train_models(X_train, y_train, X_test, y_test, random_state=42):
    """
    Entrena y evalúa múltiples modelos.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=random_state, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=random_state)
    }
    
    results = {}
    
    for name, model in models.items():
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Predecir
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calcular métricas
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)
        
        results[name] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
    
    return results

def optimize_best_model(results, X_train, y_train, cv_folds=5):
    """
    Optimiza el mejor modelo encontrado.
    """
    # Seleccionar el mejor modelo basado en R² de prueba
    best_model_name = max(results.items(), key=lambda x: x[1]['test_metrics']['R2'])[0]
    best_model = results[best_model_name]['model']
    
    # Hiperparámetros para Random Forest
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=cv_folds,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
    
    # Hiperparámetros para Gradient Boosting
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 4],
            'min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid,
            cv=cv_folds,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
    
    else:
        return best_model, "No hyperparameter tuning needed"

if __name__ == "__main__":
    print("✅ Módulo de entrenamiento de modelos cargado correctamente")