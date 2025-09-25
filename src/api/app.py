"""
API para predicción de severidad de siniestros actuariales usando FastAPI.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Severidad de Siniestros Actuariales",
    description="""API para predecir el costo esperado de siniestros de seguros 
                basado en características del asegurado y el vehículo.""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Esquemas de datos con Pydantic (validación y documentación automática)
class ClaimFeatures(BaseModel):
    """Esquema para las características de entrada del siniestro."""
    edad: int = Field(..., ge=18, le=70, description="Edad del asegurado (18-70 años)")
    tipo_vehiculo: str = Field(..., description="Tipo de vehículo: Sedan, SUV, Pickup, Motocicleta")
    antiguedad_vehiculo: int = Field(..., ge=0, le=20, description="Antigüedad del vehículo en años (0-20)")
    region: str = Field(..., description="Región: Norte, Sur, Este, Oeste")
    historial_siniestros: int = Field(..., ge=0, le=10, description="Número de siniestros previos (0-10)")

    class Config:
        schema_extra = {
            "example": {
                "edad": 45,
                "tipo_vehiculo": "SUV",
                "antiguedad_vehiculo": 5,
                "region": "Norte",
                "historial_siniestros": 2
            }
        }

class PredictionResult(BaseModel):
    """Esquema para la respuesta de predicción."""
    prediction_id: str
    timestamp: str
    log_prediction: float
    prediction: float
    confidence_interval: dict
    features: dict

class HealthCheck(BaseModel):
    """Esquema para el health check."""
    status: str
    timestamp: str
    model_version: str
    model_loaded: bool

# Cargar modelo y preprocesador
def load_artifacts():
    """Carga el modelo y preprocesador entrenados."""
    try:
        # Obtener la ruta del directorio actual
        current_dir = Path(__file__).parent
        models_dir = current_dir.parent.parent / "models"
        
        model_path = models_dir / "best_model.pkl"
        preprocessor_path = models_dir / "preprocessor.pkl"
        model_results_path = models_dir / "model_results.pkl"
        
        # Verificar que los archivos existen
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocesador no encontrado en: {preprocessor_path}")
        
        # Cargar artefactos
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Cargar resultados del modelo si existe
        model_results = {}
        if model_results_path.exists():
            model_results = joblib.load(model_results_path)
        
        logger.info("✅ Modelo y preprocesador cargados exitosamente")
        return model, preprocessor, model_results
        
    except Exception as e:
        logger.error(f"❌ Error cargando artefactos: {e}")
        raise

# Cargar artefactos al iniciar la aplicación
try:
    model, preprocessor, model_results = load_artifacts()
    MODEL_LOADED = True
    MODEL_VERSION = model_results.get('model_version', '1.0.0')
except Exception as e:
    logger.error(f"Error inicializando la aplicación: {e}")
    model, preprocessor, model_results = None, None, {}
    MODEL_LOADED = False
    MODEL_VERSION = 'none'

# Funciones auxiliares para ingeniería de características
def create_insurance_features(input_data: dict) -> pd.DataFrame:
    """
    Crea características derivadas para el input, igual que en el entrenamiento.
    """
    df = pd.DataFrame([input_data])
    
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

def transform_features(df: pd.DataFrame, preprocessor) -> np.ndarray:
    """
    Aplica el preprocesamiento a los datos de entrada.
    """
    try:
        # Transformar características principales
        processed_features = preprocessor.transform(df)
        
        # Añadir características binarias manualmente
        binary_features = df[['alto_riesgo']].values
        
        # Combinar todas las características
        final_features = np.hstack([processed_features, binary_features])
        
        return final_features
    except Exception as e:
        logger.error(f"Error en transformación de características: {e}")
        raise

# Endpoints de la API
@app.get("/", response_model=dict)
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "Bienvenido a la API de Predicción de Severidad de Siniestros Actuariales",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Endpoint para verificar el estado del servicio."""
    return {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_version": MODEL_VERSION,
        "model_loaded": MODEL_LOADED
    }

@app.post("/predict", response_model=PredictionResult)
async def predict(features: ClaimFeatures):
    """
    Predice el costo esperado de un siniestro basado en las características proporcionadas.
    
    Args:
        features: Características del asegurado y el vehículo
        
    Returns:
        PredictionResult: Predicción del costo del siniestro con metadatos
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Servicio no disponible.")
    
    try:
        # Generar ID único para la predicción
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convertir características a DataFrame
        input_dict = features.dict()
        
        # Crear características derivadas
        df_with_features = create_insurance_features(input_dict)
        
        # Aplicar transformaciones
        processed_features = transform_features(df_with_features, preprocessor)
        
        # Realizar predicción (en escala logarítmica)
        log_prediction = model.predict(processed_features)[0]
        
        # Convertir a escala original (exponenciar)
        prediction = np.expm1(log_prediction)
        
        # Calcular intervalo de confianza aproximado (para modelos de árboles)
        if hasattr(model, 'predict'):
            # Para Random Forest o Gradient Boosting, podemos usar la desviación estándar de los árboles
            if hasattr(model, 'estimators_'):
                predictions_trees = [tree.predict(processed_features)[0] for tree in model.estimators_]
                std_log = np.std(predictions_trees)
                lower_bound = np.expm1(log_prediction - 1.96 * std_log)
                upper_bound = np.expm1(log_prediction + 1.96 * std_log)
            else:
                # Para otros modelos, usar un porcentaje fijo
                margin = prediction * 0.15  # 15% de margen
                lower_bound = max(0, prediction - margin)
                upper_bound = prediction + margin
        else:
            lower_bound = prediction * 0.85
            upper_bound = prediction * 1.15
        
        logger.info(f"✅ Predicción realizada - ID: {prediction_id}, Costo: ${prediction:.2f}")
        
        return {
            "prediction_id": prediction_id,
            "timestamp": datetime.now().isoformat(),
            "log_prediction": round(log_prediction, 4),
            "prediction": round(prediction, 2),
            "confidence_interval": {
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2)
            },
            "features": input_dict
        }
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando la solicitud: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(features_list: List[ClaimFeatures]):
    """
    Predice el costo para múltiples siniestros en lote.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Servicio no disponible.")
    
    try:
        results = []
        for i, features in enumerate(features_list):
            # Reutilizar la lógica del endpoint individual
            input_dict = features.dict()
            df_with_features = create_insurance_features(input_dict)
            processed_features = transform_features(df_with_features, preprocessor)
            log_prediction = model.predict(processed_features)[0]
            prediction = np.expm1(log_prediction)
            
            results.append({
                "id": i + 1,
                "prediction": round(prediction, 2),
                "log_prediction": round(log_prediction, 4),
                "features": input_dict
            })
        
        return {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_predictions": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Error en predicción por lote: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando el lote: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Endpoint para obtener información del modelo entrenado."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    
    return {
        "model_name": model_results.get('model_name', 'Desconocido'),
        "model_version": MODEL_VERSION,
        "training_metrics": model_results.get('train_metrics', {}),
        "test_metrics": model_results.get('test_metrics', {}),
        "best_params": model_results.get('best_params', {}),
        "feature_names": model_results.get('feature_names', [])[:10]  # Primeras 10 características
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)