# 📚 Ejemplos de Uso - Pipeline Temporal

Esta carpeta contiene ejemplos prácticos de cómo usar el pipeline de entrenamiento con validación temporal y drift monitoring.

## 🚀 Quick Start

### Opción 1: Entrenamiento Básico
```bash
python src/train_pipeline_temporal.py --data WA_Fn-UseC_-HR-Employee-Attrition.csv
```

### Opción 2: Con Generación Temporal
```bash
python src/train_pipeline_temporal.py \
    --data WA_Fn-UseC_-HR-Employee-Attrition.csv \
    --temporal-gen \
    --n-months 6 \
    --scenario baseline
```

### Opción 3: Pipeline Completo (Recomendado)
```bash
python src/train_pipeline_temporal.py \
    --data WA_Fn-UseC_-HR-Employee-Attrition.csv \
    --temporal-gen \
    --n-months 6 \
    --scenario baseline \
    --temporal-val \
    --n-splits 3
```

## 📋 Parámetros Disponibles

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--data` | str | **requerido** | Nombre del archivo CSV en `data/` |
| `--temporal-gen` | flag | False | Activar generación temporal |
| `--n-months` | int | 6 | Número de meses a generar |
| `--scenario` | str | baseline | Escenario de drift |
| `--temporal-val` | flag | False | Activar validación temporal |
| `--n-splits` | int | 3 | Splits para walk-forward |

## 🎭 Escenarios de Drift Disponibles

### 1. `baseline`
- Condiciones normales de negocio
- Attrition estable (~16%)
- Sin cambios significativos

### 2. `economic_recession`
- Recesión económica
- Aumentos salariales bajos
- Mayor estrés por distancia
- Attrition aumenta gradualmente

### 3. `tech_boom`
- Boom tecnológico
- Aumentos salariales altos
- Mayor competencia por talento
- Attrition alta en roles técnicos

### 4. `high_competition`
- Alta competencia en el mercado
- Rotación acelerada
- Cambios en satisfacción laboral

## 🎯 Casos de Uso

### Caso 1: Testing de Robustez
**Objetivo:** Validar que el modelo funciona bien con datos futuros

```bash
python src/train_pipeline_temporal.py \
    --data WA_Fn-UseC_-HR-Employee-Attrition.csv \
    --temporal-gen \
    --n-months 12 \
    --scenario baseline \
    --temporal-val \
    --n-splits 4
```

**Resultado esperado:**
- ROC-AUC estable a través del tiempo
- Bajo data leakage (<0.05)
- Pocas alertas de drift

### Caso 2: Simulación de Crisis
**Objetivo:** Evaluar performance bajo condiciones adversas

```bash
python src/train_pipeline_temporal.py \
    --data WA_Fn-UseC_-HR-Employee-Attrition.csv \
    --temporal-gen \
    --n-months 12 \
    --scenario economic_recession \
    --temporal-val \
    --n-splits 4
```

**Resultado esperado:**
- Múltiples alertas de drift
- Performance decay visible
- Necesidad de reentrenamiento

### Caso 3: Desarrollo sin Datos Reales
**Objetivo:** Desarrollar features sin acceso a producción

```bash
python src/train_pipeline_temporal.py \
    --data WA_Fn-UseC_-HR-Employee-Attrition.csv \
    --temporal-gen \
    --n-months 6 \
    --scenario baseline
```

**Resultado esperado:**
- Datos sintéticos realistas
- Continuidad temporal preservada
- Privacy compliance

## 📊 Interpretación de Resultados

### Métricas de Performance
```json
{
  "roc_auc": 0.6673,
  "classification_report": {
    "1": {
      "f1-score": 0.5447,
      "precision": 0.6784,
      "recall": 0.4602
    }
  }
}
```

**Interpretación:**
- ROC-AUC > 0.65: Modelo aceptable
- F1-Score > 0.50: Balance adecuado
- Recall > 0.45: Detecta ~45% de fugas

### Drift Alerts
```json
{
  "summary": {
    "covariate_alerts": 23,
    "concept_drift_detected": false,
    "overall_status": "CRITICAL"
  }
}
```

**Interpretación:**
- 0-5 alertas: NORMAL (datos estables)
- 6-15 alertas: WARNING (monitorear)
- 16+ alertas: CRITICAL (reentrenar)

### Validación Temporal
```json
{
  "mean_metrics": {
    "roc_auc": 0.6673,
    "f1_score": 0.3602
  },
  "std_metrics": {
    "roc_auc": 0.0072,
    "f1_score": 0.1870
  }
}
```

**Interpretación:**
- Std ROC-AUC < 0.05: Performance estable
- Std F1 > 0.10: Variabilidad alta (normal en datos temporales)

## 🔍 Debugging

### Error: "No module named 'src.temporal_generator'"
**Solución:** Ejecutar desde la raíz del proyecto
```bash
cd c:/Users/FrancoYairBenko/OneDrive - IBM/Documents/Desarrollo/agentic_mlops
python src/train_pipeline_temporal.py --data ...
```

### Error: "period column not found"
**Causa:** Datos sin columna temporal
**Solución:** Usar `--temporal-gen` para generar datos con período

### Warning: "MLflow tracking failed"
**Causa:** MLflow server no está corriendo
**Solución:** 
```bash
# Iniciar MLflow
mlflow server --host 0.0.0.0 --port 5000
```

## 📁 Outputs Generados

```
models/
├── model_temporal_YYYYMMDD_HHMMSS_data_vX.joblib  # Modelo entrenado
├── metrics_temporal_YYYYMMDD_HHMMSS_data_vX.json  # Métricas detalladas
├── latest_metrics_temporal.json                    # Última ejecución
└── drift_reports/
    └── drift_report_YYYYMMDD_HHMMSS.json          # Análisis de drift
```

## 🎓 Best Practices

### ✅ DO
- Usar validación temporal para evaluar robustez
- Monitorear drift regularmente
- Generar datos sintéticos para testing
- Documentar escenarios de drift

### ❌ DON'T
- Usar datos sintéticos para inflar training set
- Ignorar alertas de drift críticas
- Mezclar datos sintéticos con reales en producción
- Confiar solo en random split

## 🔗 Referencias

- [Documentación Completa](../docs/IMPLEMENTATION_GUIDE.md)
- [Análisis Crítico](../docs/propuesta_analisis_critico.md)
- [Quick Start](../docs/QUICK_START.md)
- [Experimentos](../experiments/README.md)

## 💡 Tips para Entrevistas

**Pregunta:** "¿Cómo validarías un modelo de ML en producción?"

**Respuesta:**
> "Implementé un sistema de validación temporal con walk-forward validation que respeta el orden cronológico de los datos. Uso datos sintéticos con continuidad temporal para simular escenarios de drift y evaluar la robustez del modelo. Monitoreo drift con PSI, KS-test y Wasserstein distance, y reentrenamos cuando detectamos más de 15 alertas críticas."

**Pregunta:** "¿Qué harías si detectas drift en producción?"

**Respuesta:**
> "Primero clasifico el tipo de drift: covariate shift (cambios en P(X)) o concept drift (cambios en P(Y|X)). Para covariate shift, evalúo si el modelo es robusto con validación temporal. Si hay concept drift, reentrenamos con datos recientes. Uso métricas como PSI para cuantificar la severidad y decidir si es necesario reentrenar inmediatamente o esperar más datos."