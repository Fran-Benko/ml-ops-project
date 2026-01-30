# Script para ejecutar experimento de comparacion con Podman en Windows
# Autor: IBM Bob (Data Science Expert)
# Fecha: 2026-01-20

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "EXPERIMENTO: Comparacion de Generadores (Podman)" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

# Verificar que podman esta instalado
try {
    $podmanVersion = podman --version
    Write-Host "OK Podman detectado: $podmanVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Podman no esta instalado" -ForegroundColor Red
    Write-Host "   Instalar desde: https://podman.io/getting-started/installation" -ForegroundColor Yellow
    exit 1
}

# Obtener directorio actual
$currentDir = Get-Location

# Construir imagen
Write-Host ""
Write-Host "Construyendo imagen..." -ForegroundColor Yellow
podman build -t mlops_experiment:latest -f experiments/Dockerfile .

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR al construir imagen" -ForegroundColor Red
    exit 1
}

# Crear directorios necesarios
Write-Host ""
Write-Host "Creando directorios..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "experiments\results" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null

# Ejecutar contenedor
Write-Host ""
Write-Host "Ejecutando experimento..." -ForegroundColor Yellow
Write-Host "   (Esto puede tomar varios minutos...)" -ForegroundColor Gray

podman run --rm `
    --name mlops_experiment `
    -v "${currentDir}\data:/app/data:ro,Z" `
    -v "${currentDir}\src:/app/src:ro,Z" `
    -v "${currentDir}\experiments:/app/experiments:Z" `
    -v "${currentDir}\models:/app/models:Z" `
    -e PYTHONUNBUFFERED=1 `
    mlops_experiment:latest

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR al ejecutar experimento" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Experimento completado exitosamente" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Resultados guardados en:" -ForegroundColor Yellow
Write-Host "   - experiments\comparison_results.json" -ForegroundColor White
Write-Host "   - models\drift_report_*.json" -ForegroundColor White
Write-Host ""

# Made with Bob
