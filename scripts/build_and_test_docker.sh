#!/bin/bash

# Script para construir y probar la imagen Docker
set -e

echo "🚀 Iniciando construcción de la imagen Docker..."

# Nombre de la imagen
IMAGE_NAME="mlops-actuarial-api"
CONTAINER_NAME="actuarial-api-test"

# Construir la imagen
echo "📦 Construyendo imagen Docker..."
docker build -t $IMAGE_NAME .

echo "✅ Imagen construida exitosamente"

# Verificar que la imagen se creó
echo "🔍 Verificando imagen..."
docker images | grep $IMAGE_NAME

# Ejecutar contenedor en segundo plano
echo "🐳 Iniciando contenedor..."
docker run -d --name $CONTAINER_NAME -p 8000:8000 $IMAGE_NAME

# Esperar a que la aplicación esté lista
echo "⏳ Esperando a que la aplicación inicie..."
sleep 10

# Probar health check
echo "🔍 Probando health check..."
curl -f http://localhost:8000/health || {
    echo "❌ Health check falló"
    docker logs $CONTAINER_NAME
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    exit 1
}

echo "✅ Health check exitoso"

# Probar predicción
echo "🎯 Probando predicción..."
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 45,
    "tipo_vehiculo": "SUV",
    "antiguedad_vehiculo": 5,
    "region": "Norte",
    "historial_siniestros": 2
  }' || {
    echo "❌ Prueba de predicción falló"
    docker logs $CONTAINER_NAME
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    exit 1
}

echo "✅ Prueba de predicción exitosa"

# Detener y eliminar contenedor
echo "🧹 Limpiando contenedor..."
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo "🎉 ¡Todas las pruebas pasaron! La imagen Docker está lista.