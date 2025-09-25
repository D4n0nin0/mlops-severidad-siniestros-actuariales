"""
Módulo para ingeniería de características y preprocesamiento.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Función nombrada para reemplazar la lambda - ¡IMPORTANTE PARA PICKLING!
def identity_transform(x):
    """
    Transformación identidad para características binarias.
    Serializable con pickle/joblib.
    """
    return x

def create_insurance_features(df):
    """
    Crea características derivadas específicas para seguros.
    """
    df = df.copy()
    
    # 1. Edad agrupada (rangos actuariales)
    df['edad_grupo'] = pd.cut(df['edad'], 
                             bins=[18, 25, 35, 45, 55, 65, 70],
                             labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-70'])
    
    # 2. Antigüedad agrupada
    df['antiguedad_grupo'] = pd.cut(df['antiguedad_vehiculo'],
                                   bins=[0, 3, 7, 15, 20],
                                   labels=['0-3', '4-7', '8-15', '16-20'])
    
    # 3. Variable binaria: conductor de alto riesgo
    df['alto_riesgo'] = ((df['edad'] < 25) | (df['edad'] > 65) | 
                        (df['historial_siniestros'] >= 3)).astype(int)
    
    return df

def create_feature_pipeline():
    """
    Crea un pipeline de preprocesamiento completo para características de seguros.
    """
    # Columnas originales
    numeric_features = ['edad', 'antiguedad_vehiculo', 'historial_siniestros']
    categorical_features = ['tipo_vehiculo', 'region']
    
    # Columnas derivadas
    categorical_derived = ['edad_grupo', 'antiguedad_grupo']
    binary_features = ['alto_riesgo']
    
    # Pipelines individuales
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # USAR FUNCIÓN NOMBRADA EN LUGAR DE LAMBDA
    binary_transformer = FunctionTransformer(identity_transform, validate=False)
    
    # Preprocesador completo
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('cat_derived', categorical_transformer, categorical_derived),
            ('binary', binary_transformer, binary_features)
        ])
    
    return preprocessor

def prepare_data(df, test_size=0.2, random_state=42):
    """
    Prepara los datos para modelado.
    """
    y = np.log1p(df['costo_siniestro'])
    X = df.drop('costo_siniestro', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=X['region']
    )
    
    return X_train, X_test, y_train, y_test