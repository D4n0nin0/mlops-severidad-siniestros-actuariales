"""
Módulo para la carga y preparación inicial de datos.
"""
import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga datos desde un archivo CSV.
    
    Args:
        file_path (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Datos cargados correctamente. Dimensiones: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {file_path}")
        return None

def generate_sample_data() -> pd.DataFrame:
    """
    Genera datos de muestra para pruebas (datos sintéticos de seguros).
    
    Returns:
        pd.DataFrame: DataFrame con datos de ejemplo
    """
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'edad': np.random.randint(18, 70, n_samples),
        'tipo_vehiculo': np.random.choice(['Sedan', 'SUV', 'Pickup', 'Motocicleta'], n_samples),
        'antiguedad_vehiculo': np.random.randint(0, 20, n_samples),
        'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], n_samples),
        'historial_siniestros': np.random.randint(0, 5, n_samples),
        'costo_siniestro': np.random.gamma(2, 1000, n_samples)  # Distribución gamma típica en seguros
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Ejemplo de uso
    df = generate_sample_data()
    print("Datos de muestra generados:")
    print(df.head())