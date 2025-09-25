"""
Módulo para ingeniería de características y preprocesamiento.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_feature_pipeline() -> Pipeline:
    """
    Crea un pipeline de preprocesamiento para características numéricas y categóricas.
    
    Returns:
        Pipeline: Pipeline de preprocesamiento
    """
    # Definir columnas numéricas y categóricas
    numeric_features = ['edad', 'antiguedad_vehiculo', 'historial_siniestros']
    categorical_features = ['tipo_vehiculo', 'region']
    
    # Crear transformadores
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    # Combinar transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

if __name__ == "__main__":
    # Ejemplo de uso
    pipeline = create_feature_pipeline()
    print("Pipeline de características creado exitosamente")