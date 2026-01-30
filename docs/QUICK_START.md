# ⚡ Quick Start: Generación Sintética Temporal

Guía rápida para empezar a usar el sistema de generación temporal en 5 minutos.

---

## 🚀 Instalación Rápida

```bash
# 1. Clonar repositorio (si aún no lo tienes)
git clone <repo-url>
cd agentic_mlops

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Verificar instalación
python -c "from src.temporal_generator import TemporalHRGenerator; print('✅ OK')"
```

---

## 📝 Ejemplo 1: Generar Datos Temporales

```python
from src.temporal_generator import TemporalHRGenerator
import pandas as pd

# Cargar datos semilla
seed_data = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Inicializar generador
generator = TemporalHRGenerator(seed_data, start_date="2024-01-01")

# Generar 6 meses de datos
temporal_data = generator.generate_temporal_sequence(
    n_months=6,
    retention_rate=0.85  # 85% de empleados permanecen
)

# Guardar
temporal_data.to_csv('data/temporal_sequence.csv', index=False)
print(f"✅ Generados {len(temporal_data)} registros en {temporal_data['DataMonth'].nunique()} meses")
```

**Salida esperada:**
```
✅ TemporalHRGenerator inicializado con 1470 empleados
📅 Fecha inicial: 2024-01-01

🔄 Generando Mes 1 (2024-02)
   👥 Empleados retenidos: 1249
   🆕 Nuevos empleados: 221
   ✅ Total empleados: 1470
   📊 Tasa de attrition: 15.03%

...

✅ Secuencia temporal generada: 7 meses, 10290 registros
```

---

## 📊 Ejemplo 2: Detectar Drift

```python
from src.drift_monitor import DriftMonitor

# Datos de referencia (mes 1)
reference = temporal_data[temporal_data['DataMonth'] == temporal_data['DataMonth'].min()]

# Datos nuevos (mes 6)
new_data = temporal_data[temporal_data['DataMonth'] == temporal_data['DataMonth'].max()]

# Inicializar monitor
monitor = DriftMonitor(reference, target_col='Attrition')

# Generar reporte
report = monitor.generate_drift_report(
    new_data,
    save_path='models/drift_report.json'
)

# Ver top features con drift
top_drifted = monitor.get_top_drifted_features(report, top_n=5)
for feature, score, severity in top_drifted:
    print(f"  {feature}: PSI={score:.4f} ({severity})")
```

**Salida esperada:**
```
🔍 Detectando Covariate Shift...
   ⚠️ Alertas detectadas: 3/25 features

📊 RESUMEN:
   Features monitoreadas: 25
   Alertas de covariate shift: 3
   Concept drift detectado: NO
   Estado general: WARNING

💾 Reporte guardado en: models/drift_report.json
```

---

## 🔄 Ejemplo 3: Validación Temporal

```python
from src.temporal_validation import TemporalValidator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Crear modelo simple
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])

# Inicializar validador
validator = TemporalValidator(date_column='DataMonth', target_column='Attrition')

# Walk-forward validation
results = validator.walk_forward_validation(
    temporal_data,
    model,
    n_splits=4,
    strategy='expanding'
)

print(f"ROC-AUC promedio: {results['mean_metrics']['roc_auc']:.4f}")
print(f"F1-Score promedio: {results['mean_metrics']['f1_score']:.4f}")
```

**Salida esperada:**
```
🔄 Walk-Forward Validation (expanding window)
   Splits: 4

   Split 1/4:
      Train: 2 períodos (2940 registros)
      Test: 2024-04 (1470 registros)
      ROC-AUC: 0.8234
      F1-Score: 0.4521

...

📊 RESUMEN:
   ROC-AUC: 0.8156 ± 0.0234
   F1-Score: 0.4389 ± 0.0312
```

---

## 🆚 Ejemplo 4: Comparar Generadores

```bash
# Ejecutar comparación completa
cd experiments
python compare_generators.py
```

**Salida esperada:**
```
======================================================================
COMPARACIÓN COMPLETA: GENERADOR ORIGINAL VS TEMPORAL
======================================================================

GENERADOR ORIGINAL (Batches Independientes)
======================================================================
🎲 Generando datos bajo escenario: Normal Operation
✅ Datos generados: (1200, 32)
   Batches: 3

GENERADOR TEMPORAL (Continuidad de Cohortes)
======================================================================
✅ TemporalHRGenerator inicializado con 1470 empleados
...

======================================================================
RESUMEN DE COMPARACIÓN
======================================================================

📊 PERFORMANCE (Random Split):
   Original: ROC-AUC = 0.8456
   Temporal: ROC-AUC = 0.8234
   Diferencia: -0.0222

📊 PERFORMANCE (Temporal Validation):
   Original: ROC-AUC = 0.7234
   Temporal: ROC-AUC = 0.8156
   Diferencia: +0.0922

⚠️ DATA LEAKAGE DETECTION:
   Original: 0.1222 🚨 ALTO
   Temporal: 0.0078 ✅ BAJO

🔬 REALISMO DE DATOS:
   Temporal - Continuidad de empleados: 82.34%
   Original - Volatilidad attrition: 0.2341
   Temporal - Volatilidad attrition: 0.0456

======================================================================
VEREDICTO FINAL
======================================================================
✅ Temporal gana en: Menor data leakage
✅ Temporal gana en: Continuidad de empleados
✅ Temporal gana en: Estabilidad de attrition
✅ Temporal gana en: Performance con validación temporal

🏆 SCORE FINAL:
   Original: 0 puntos
   Temporal: 7 puntos

🎯 GANADOR: TEMPORAL

💾 Resultados guardados en: experiments/comparison_results.json
```

---

## 🎯 Casos de Uso Comunes

### Caso 1: Testing de Pipeline de Retraining
```python
# Simular 12 meses de datos
temporal_data = generator.generate_temporal_sequence(n_months=12)

# Entrenar en primeros 6 meses
train_data = temporal_data[temporal_data['DataMonth'] <= '2024-06']
model.fit(train_data)

# Evaluar en siguientes 6 meses
for month in temporal_data['DataMonth'].unique()[6:]:
    test_data = temporal_data[temporal_data['DataMonth'] == month]
    score = model.score(test_data)
    print(f"{month}: ROC-AUC = {score:.4f}")
```

### Caso 2: Simulación de Escenarios de Drift
```python
# Escenario: Recesión económica
drift_params = {
    'salary_increase': -0.05,  # -5% salarios
    'satisfaction_decay': -0.20,  # -20% satisfacción
    'overtime_increase': 0.15  # +15% overtime
}

recession_data = generator.generate_next_month(
    retention_rate=0.90,  # Menos gente se va
    drift_params=drift_params,
    scenario='recession'
)

# Evaluar impacto en modelo
impact = model.score(recession_data)
print(f"Performance en recesión: {impact:.4f}")
```

### Caso 3: Detección Automática de Retraining
```python
# Monitorear drift mensualmente
for month in temporal_data['DataMonth'].unique()[1:]:
    new_data = temporal_data[temporal_data['DataMonth'] == month]
    
    # Detectar drift
    report = monitor.generate_drift_report(new_data, model=model)
    
    # Trigger retraining si hay drift significativo
    if report['summary']['overall_status'] == 'CRITICAL':
        print(f"🚨 ALERTA: Retraining requerido en {month}")
        # Aquí iría lógica de retraining
```

---

## 📚 Próximos Pasos

1. **Leer documentación completa:** [`docs/IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md)
2. **Revisar análisis crítico:** [`docs/propuesta_analisis_critico.md`](propuesta_analisis_critico.md)
3. **Explorar código fuente:** [`src/temporal_generator.py`](../src/temporal_generator.py)
4. **Ejecutar comparación:** `python experiments/compare_generators.py`

---

## 🆘 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'src'"
```bash
# Solución: Agregar directorio raíz al PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Error: "FileNotFoundError: data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
```bash
# Solución: Verificar que estás en el directorio raíz
cd agentic_mlops
ls data/  # Debe mostrar el archivo CSV
```

### Warning: "Data leakage detected"
```python
# Solución: Usar validación temporal en vez de random split
# ❌ NO HACER:
train_test_split(X, y, test_size=0.2)

# ✅ HACER:
validator.walk_forward_validation(data, model, strategy='expanding')
```

---

## 💡 Tips y Trucos

1. **Ajustar retention_rate según industria:**
   - Tech: 0.80-0.85 (alta rotación)
   - Banca: 0.90-0.95 (baja rotación)
   - Retail: 0.70-0.80 (muy alta rotación)

2. **Drift gradual vs abrupto:**
   - Gradual: `{'salary_increase': 0.02}` por mes
   - Abrupto: `{'satisfaction_decay': -0.50}` en un mes

3. **Validar realismo:**
   ```python
   # Verificar continuidad de empleados
   unique_employees = temporal_data['EmployeeNumber'].nunique()
   total_records = len(temporal_data)
   continuity = 1 - (unique_employees / total_records)
   print(f"Continuidad: {continuity:.2%}")  # Debe ser > 70%
   ```

---

**¿Preguntas?** Consulta la [guía completa](IMPLEMENTATION_GUIDE.md) o abre un issue en GitHub.