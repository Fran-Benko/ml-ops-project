#!/bin/bash
# Script para ejecutar con Podman
echo "🚀 Iniciando Agentic MLOps con Podman..."

# Construir e iniciar
podman-compose up --build -d

echo "✅ Aplicación corriendo en http://localhost:8501"
echo "Para ver logs: podman logs -f agentic-mlops-app"
