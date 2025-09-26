#!/usr/bin/env python3
"""
Script de pruebas para CI/CD
"""
import sys
import os
import importlib.util

def test_imports():
    """Test que todos los módulos pueden importarse."""
    modules_to_test = [
        'src.data.make_dataset',
        'src.features.build_features', 
        'src.models.train_model',
        'src.api.app'
    ]
    
    for module_path in modules_to_test:
        try:
            # Convertir ruta de módulo a ruta de archivo
            file_path = module_path.replace('.', '/') + '.py'
            spec = importlib.util.spec_from_file_location(module_path, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✅ {module_path} imported successfully")
        except Exception as e:
            print(f"❌ Failed to import {module_path}: {e}")
            return False
    return True

def test_data_generation():
    """Test la generación de datos."""
    try:
        from src.data.make_dataset import generate_sample_data
        df = generate_sample_data()
        assert df.shape[0] == 1000, f"Expected 1000 samples, got {df.shape[0]}"
        assert 'costo_siniestro' in df.columns, "Missing target column"
        print("✅ Data generation test passed")
        return True
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_feature_pipeline():
    """Test el pipeline de características."""
    try:
        from src.features.build_features import create_feature_pipeline
        pipeline = create_feature_pipeline()
        assert pipeline is not None, "Pipeline should not be None"
        print("✅ Feature pipeline test passed")
        return True
    except Exception as e:
        print(f"❌ Feature pipeline test failed: {e}")
        return False

def test_api_schema():
    """Test los esquemas de la API."""
    try:
        from src.api.app import ClaimFeatures
        sample_data = {
            'edad': 45,
            'tipo_vehiculo': 'SUV',
            'antiguedad_vehiculo': 5,
            'region': 'Norte',
            'historial_siniestros': 2
        }
        claim = ClaimFeatures(**sample_data)
        print("✅ API schema test passed")
        return True
    except Exception as e:
        print(f"❌ API schema test failed: {e}")
        return False

def main():
    """Ejecutar todas las pruebas."""
    print("🚀 Running CI/CD tests...")
    
    tests = [
        test_imports,
        test_data_generation,
        test_feature_pipeline,
        test_api_schema
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()