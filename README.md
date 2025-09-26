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