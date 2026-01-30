# Script para ejecutar tests en contenedor Podman
# ================================================
# Ejecuta todos los tests en un ambiente aislado

Write-Host "🧪 Ejecutando Tests en Contenedor Podman" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Construir imagen de testing
Write-Host "`n📦 Construyendo imagen de testing..." -ForegroundColor Yellow
podman build -t agentic-mlops-test:latest -f Dockerfile.test .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error construyendo imagen" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Imagen construida exitosamente" -ForegroundColor Green

# Ejecutar tests
Write-Host "`n🧪 Ejecutando tests..." -ForegroundColor Yellow
podman run --rm `
    -v ${PWD}/htmlcov:/app/htmlcov `
    agentic-mlops-test:latest

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Algunos tests fallaron" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ Todos los tests pasaron exitosamente!" -ForegroundColor Green
Write-Host "📊 Reporte de coverage generado en: htmlcov/index.html" -ForegroundColor Cyan

# Made with Bob
