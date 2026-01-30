# Script de ejemplo para entrenar con validación temporal (Windows)
# Autor: Franco Benko

Write-Host "========================================================================"
Write-Host "EJEMPLO: Pipeline de Entrenamiento con Validación Temporal"
Write-Host "========================================================================"

# 1. Entrenamiento básico (sin generación temporal)
Write-Host ""
Write-Host "1️⃣ Entrenamiento básico con datos originales..." -ForegroundColor Cyan
python src/train_pipeline_temporal.py `
    --data WA_Fn-UseC_-HR-Employee-Attrition.csv

# 2. Entrenamiento con generación temporal
Write-Host ""
Write-Host "2️⃣ Entrenamiento con generación temporal (6 meses)..." -ForegroundColor Cyan
python src/train_pipeline_temporal.py `
    --data WA_Fn-UseC_-HR-Employee-Attrition.csv `
    --temporal-gen `
    --n-months 6 `
    --scenario baseline

# 3. Entrenamiento con generación temporal + validación temporal
Write-Host ""
Write-Host "3️⃣ Entrenamiento completo (generación + validación temporal)..." -ForegroundColor Cyan
python src/train_pipeline_temporal.py `
    --data WA_Fn-UseC_-HR-Employee-Attrition.csv `
    --temporal-gen `
    --n-months 6 `
    --scenario baseline `
    --temporal-val `
    --n-splits 3

# 4. Escenario de recesión económica
Write-Host ""
Write-Host "4️⃣ Simulación de recesión económica..." -ForegroundColor Cyan
python src/train_pipeline_temporal.py `
    --data WA_Fn-UseC_-HR-Employee-Attrition.csv `
    --temporal-gen `
    --n-months 12 `
    --scenario economic_recession `
    --temporal-val `
    --n-splits 4

Write-Host ""
Write-Host "========================================================================"
Write-Host "✅ Ejemplos completados" -ForegroundColor Green
Write-Host "========================================================================"
Write-Host "Revisa los resultados en:"
Write-Host "  - models/latest_metrics_temporal.json"
Write-Host "  - models/drift_reports/"
Write-Host "  - MLflow UI: http://localhost:5000"

# Made with Bob
