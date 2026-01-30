# Script para ejecutar pipeline temporal en Podman (Windows)
# Autor: Franco Benko

param(
    [string]$Mode = "basic",
    [int]$Months = 3,
    [string]$Scenario = "baseline"
)

Write-Host "========================================================================"
Write-Host "PIPELINE TEMPORAL EN PODMAN"
Write-Host "========================================================================"

# Verificar Podman
$podmanVersion = podman --version 2>$null
if (-not $podmanVersion) {
    Write-Host "Error: Podman no esta instalado" -ForegroundColor Red
    exit 1
}
Write-Host "Podman detectado: $podmanVersion" -ForegroundColor Green

# Construir imagen
Write-Host ""
Write-Host "Construyendo imagen..." -ForegroundColor Cyan
podman build -t mlops_temporal:latest -f examples/Dockerfile .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error al construir imagen" -ForegroundColor Red
    exit 1
}

# Crear directorio para outputs si no existe
if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
}
if (-not (Test-Path "models/drift_reports")) {
    New-Item -ItemType Directory -Path "models/drift_reports" | Out-Null
}

# Ejecutar según modo
Write-Host ""
Write-Host "Ejecutando pipeline en modo: $Mode" -ForegroundColor Cyan

switch ($Mode) {
    "basic" {
        Write-Host "   Modo: Entrenamiento basico (sin generacion temporal)"
        podman run --rm `
            -v ${PWD}/models:/app/models:Z `
            mlops_temporal:latest `
            python src/train_pipeline_temporal.py `
            --data WA_Fn-UseC_-HR-Employee-Attrition.csv
    }
    "temporal" {
        $msg = "   Modo: Con generacion temporal ({0} meses, escenario: {1})" -f $Months, $Scenario
        Write-Host $msg -ForegroundColor Yellow
        podman run --rm `
            -v ${PWD}/models:/app/models:Z `
            mlops_temporal:latest `
            python src/train_pipeline_temporal.py `
            --data WA_Fn-UseC_-HR-Employee-Attrition.csv `
            --temporal-gen `
            --n-months $Months `
            --scenario $Scenario
    }
    "full" {
        $msg = "   Modo: Pipeline completo (generacion + validacion temporal, {0} meses)" -f $Months
        Write-Host $msg -ForegroundColor Yellow
        podman run --rm `
            -v ${PWD}/models:/app/models:Z `
            mlops_temporal:latest `
            python src/train_pipeline_temporal.py `
            --data WA_Fn-UseC_-HR-Employee-Attrition.csv `
            --temporal-gen `
            --n-months $Months `
            --scenario $Scenario `
            --temporal-val `
            --n-splits 3
    }
    default {
        Write-Host "Modo invalido: $Mode" -ForegroundColor Red
        Write-Host "Modos disponibles: basic, temporal, full"
        exit 1
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "Pipeline completado exitosamente" -ForegroundColor Green
    Write-Host "========================================================================"
    Write-Host "Resultados guardados en:"
    Write-Host "  - models/latest_metrics_temporal.json"
    Write-Host "  - models/drift_reports/"
} else {
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "Pipeline fallo" -ForegroundColor Red
    Write-Host "========================================================================"
    exit 1
}

# Made with Bob
