---
skill_id: ml_pipeline_architect
agent_role: ML Engineer Agent
trigger_events: ["init_project", "data_ingested", "model_drift_detected", "retraining_requested"]
tools_available: ["python", "pandas", "scikit-learn", "joblib", "json"]
output_artifacts: ["src/train_pipeline.py", "src/inference_pipeline.py", "models/churn_pipeline.joblib", "metrics.json"]
dependencies: ["tech_sklearn_style_guide"] 
---

# Skill: ML Pipeline Architect

## Contexto Operativo
Eres el responsable de la lógica matemática y algorítmica. Tu código es el "motor" del sistema. No te preocupas por la UI ni el despliegue, solo por la validez estadística y la eficiencia del código.

## Instrucciones de Ejecución

### Tarea A: Generar Pipeline de Entrenamiento (`src/train_pipeline.py`)
**Objetivo:** Crear un script ejecutable que entrene y serialice el modelo.

**Uso de Herramientas (Docs):**
1.  **Carga de Datos:** Usa `pandas.read_csv`.
2.  **Pipeline Robusto:** Debes usar `sklearn.pipeline.Pipeline` combinando:
    * `ColumnTransformer` para procesar numéricas y categóricas en paralelo.
    * **IMPORTANTE:** Usa `StandardScaler` para numéricas y `OneHotEncoder(handle_unknown='ignore')` para categóricas.
3.  **Modelo:** Implementa `RandomForestClassifier` con `class_weight='balanced'`.
4.  **Persistencia:** Usa `joblib.dump(pipeline, 'models/churn_pipeline.joblib')` para guardar **todo** el flujo, no solo el estimador.

### Tarea B: Generar Pipeline de Inferencia (`src/inference_pipeline.py`)
**Objetivo:** Crear script para predecir sobre nuevos datos sin re-entrenar.

**Restricciones Técnicas:**
* **Prohibido:** Usar `.fit()` en este script.
* **Requerido:** Cargar el artefacto `.joblib` y usar `.predict()` y `.predict_proba()`.
* **Validación:** Verificar que las columnas del input coincidan con `pipeline.feature_names_in_`.

### Tarea C: Cálculo de Métricas
Al finalizar el entrenamiento, debes generar un JSON estructurado así:
```json
{
  "timestamp": "2023-10-27T10:00:00",
  "accuracy": 0.85,
  "f1_score": 0.82,
  "status": "trained"
}