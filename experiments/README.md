# 🧪 Experimentos: Comparación de Generadores

Este directorio contiene el experimento de comparación entre el generador original y el generador temporal mejorado.

---

## 📋 Contenido

- `compare_generators.py` - Script principal de comparación
- `Dockerfile` - Imagen Docker para ejecutar experimentos
- `docker-compose.yml` - Orquestación con Docker Compose
- `run_experiment.ps1` - Script de ejecución para Windows (Podman)
- `run_experiment.sh` - Script de ejecución para Linux/Mac (Podman)

---

## 🚀 Ejecución del Experimento

### Opción 1: Con Podman (Recomendado)

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File experiments\run_experiment.ps1
```

**Linux/Mac:**
```bash
chmod +x experiments/run_experiment.sh
./experiments/run_experiment.sh
```

### Opción 2: Con Docker

```bash
cd experiments
docker-compose up --build
```

### Opción 3: Ejecución Manual (Sin contenedor)

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar comparación
python experiments/compare_generators.py
```

---

## 📊 Resultados Esperados

El experimento genera los siguientes archivos:

1. **`experiments/comparison_results.json`** - Resultados completos de la comparación
2. **`models/drift_report_*.json`** - Reportes de drift detection

### Estructura del Resultado

```json
{
  "original_generator": {
    "random_split": { "roc_auc": 0.85, "f1_score": 0.45 },
    "temporal_validation": { "mean_metrics": {...} },
    "drift_analysis": {...},
    "realism": {...}
  },
  "temporal_generator": {
    "random_split": { "roc_auc": 0.82, "f1_score": 0.43 },
    "temporal_validation": { "mean_metrics": {...} },
    "drift_analysis": {...},
    "realism": { "employee_continuity": 0.85 }
  },
  "comparison": {
    "winner": "TEMPORAL",
    "scores": { "original": 2, "temporal": 7 }
  }
}
```

---

## 🎯 Métricas Evaluadas

### 1. Performance del Modelo
- **Random Split:** ROC-AUC, F1-Score, Precision, Recall
- **Temporal Validation:** Walk-forward expanding
- **Data Leakage Detection:** Diferencia entre random y temporal

### 2. Análisis de Drift
- **PSI (Population Stability Index):** Por feature
- **KS-Test:** Cambios en distribuciones
- **Número de alertas:** Features con drift significativo

### 3. Realismo de Datos
- **Continuidad de empleados:** % de empleados que persisten
- **Estabilidad de attrition:** Volatilidad entre períodos
- **Distribución de edad:** Desviación vs datos reales

---

## 📈 Interpretación de Resultados

### Criterios de Evaluación

| Criterio | Peso | Descripción |
|----------|------|-------------|
| Data Leakage | 2 pts | Menor diferencia random vs temporal |
| Continuidad | 2 pts | Mayor continuidad de empleados |
| Estabilidad | 1 pt | Menor volatilidad de attrition |
| Performance | 2 pts | Mayor ROC-AUC en validación temporal |

### Veredicto

El generador con **mayor puntaje total** es el ganador.

**Esperado:** Temporal gana con 7 puntos vs 2 del original

---

## 🔍 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'pandas'"

**Solución:** Instalar dependencias
```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError: data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

**Solución:** Verificar que estás en el directorio raíz
```bash
cd agentic_mlops
ls data/  # Debe mostrar el archivo CSV
```

### Error: Podman no está instalado

**Solución:** Instalar Podman
- Windows: https://podman.io/getting-started/installation
- Linux: `sudo apt install podman` o `sudo yum install podman`
- Mac: `brew install podman`

### El experimento tarda mucho

**Normal:** El experimento puede tardar 5-10 minutos dependiendo de tu hardware.

Incluye:
- Generación de datos sintéticos (2 generadores)
- Entrenamiento de modelos (múltiples splits)
- Análisis de drift (25+ features)
- Cálculo de métricas de realismo

---

## 📚 Documentación Relacionada

- **Análisis Crítico:** [`docs/propuesta_analisis_critico.md`](../docs/propuesta_analisis_critico.md)
- **Guía de Implementación:** [`docs/IMPLEMENTATION_GUIDE.md`](../docs/IMPLEMENTATION_GUIDE.md)
- **Quick Start:** [`docs/QUICK_START.md`](../docs/QUICK_START.md)

---

## 🤝 Contribuciones

Para agregar nuevos experimentos:

1. Crear nuevo script en `experiments/`
2. Agregar al `Dockerfile` si requiere dependencias adicionales
3. Documentar en este README
4. Actualizar `docker-compose.yml` si es necesario

---

**Última actualización:** 2026-01-20  
**Autor:** IBM Bob (Data Science Expert)