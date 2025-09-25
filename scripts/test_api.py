"""
Script para probar la API de predicción localmente.
"""
import requests
import json
import time

# Configuración
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Prueba el endpoint de health check."""
    print("🔍 Probando health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_single_prediction():
    """Prueba una predicción individual."""
    print("\n🎯 Probando predicción individual...")
    
    data = {
        "edad": 45,
        "tipo_vehiculo": "SUV",
        "antiguedad_vehiculo": 5,
        "region": "Norte",
        "historial_siniestros": 2
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ Predicción exitosa:")
            print(f"   ID: {result['prediction_id']}")
            print(f"   Costo predicho: ${result['prediction']:,.2f}")
            print(f"   Intervalo de confianza: ${result['confidence_interval']['lower_bound']:,.2f} - ${result['confidence_interval']['upper_bound']:,.2f}")
        else:
            print(f"❌ Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_prediction():
    """Prueba predicción por lote."""
    print("\n📦 Probando predicción por lote...")
    
    batch_data = [
        {
            "edad": 25,
            "tipo_vehiculo": "Sedan",
            "antiguedad_vehiculo": 2,
            "region": "Sur",
            "historial_siniestros": 0
        },
        {
            "edad": 70,
            "tipo_vehiculo": "Pickup",
            "antiguedad_vehiculo": 15,
            "region": "Este",
            "historial_siniestros": 5
        },
        {
            "edad": 35,
            "tipo_vehiculo": "Motocicleta",
            "antiguedad_vehiculo": 3,
            "region": "Oeste",
            "historial_siniestros": 1
        }
    ]
    
    try:
        response = requests.post(f"{BASE_URL}/batch_predict", json=batch_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Lote procesado: {result['total_predictions']} predicciones")
            for pred in result['results']:
                print(f"   - ID {pred['id']}: ${pred['prediction']:,.2f}")
        else:
            print(f"❌ Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Prueba el endpoint de información del modelo."""
    print("\n📊 Probando información del modelo...")
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Modelo: {info['model_name']}")
            print(f"   Versión: {info['model_version']}")
            print(f"   R² en prueba: {info['test_metrics'].get('R2', 'N/A')}")
        else:
            print(f"❌ Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Iniciando pruebas de la API...")
    
    # Esperar un poco para que la API esté lista
    time.sleep(2)
    
    tests = [
        test_health_check,
        test_single_prediction,
        test_batch_prediction,
        test_model_info
    ]
    
    results = []
    for test in tests:
        success = test()
        results.append(success)
        time.sleep(1)  # Pequeña pausa entre tests
    
    print(f"\n📈 Resumen de pruebas: {sum(results)}/{len(results)} exitosas")
    
    if all(results):
        print("🎉 ¡Todas las pruebas pasaron!")
    else:
        print("⚠️  Algunas pruebas fallaron")