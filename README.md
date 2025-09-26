# 🏥 MLOps para Severidad de Siniestros Actuariales

![CI/CD](https://github.com/TU_USUARIO/mlops-severidad-siniestros-actuariales/actions/workflows/ci-cd.yml/badge.svg)
![Tests](https://github.com/TU_USUARIO/mlops-severidad-siniestros-actuariales/actions/workflows/tests.yml/badge.svg)
![Docker](https://github.com/TU_USUARIO/mlops-severidad-siniestros-actuariale/actions/workflows/docker.yml/badge.svg)

## 📊 Descripción del Proyecto

Sistema completo de MLOps para predecir la severidad de siniestros de seguros...

## 🚀 Estado del CI/CD

El proyecto implementa una pipeline completa de CI/CD que incluye:

- ✅ **Pruebas automatizadas** de código Python
- ✅ **Construcción y testing** de imagen Docker  
- ✅ **Validación** de notebooks y estructura
- ✅ **Notificaciones** de estado de builds

## 🔄 Pipeline de CI/CD

```mermaid
graph LR
    A[Push/PR] --> B[Lint & Test Python]
    B --> C[Build Docker Image]
    C --> D[Test Docker Container]
    D --> E[Notify Results]
```

# 🏥 MLOps - Sistema de Predicción de Severidad de Siniestros Actuariales

Este proyecto implementa un sistema de machine learning para predecir la severidad de siniestros actuariales con un pipeline completo de MLOps.

## 🚀 Ejecución Rápida

### Opción 1: Usar el script automático (Recomendado)

```bash
# Dar permisos de ejecución al script
chmod +x run_app.sh

# Ejecutar la aplicación
./run_app.sh
```

### Opción 2: Ejecucion manual paso a paso

# 1. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la aplicación
python -m src.api.app

# 🌐 Acceso a la Aplicación

Una vez ejecutado el programa, accede a:

    API Documentation: http://localhost:8000/docs

    Interfaz de la API: http://localhost:8000

    Health Check: http://localhost:8000/health

## 🐳 Ejecución con Docker

### Construir la imagen
docker build -t actuarial-mlops .

### Ejecutar el contenedor
docker run -p 8000:8000 actuarial-mlops