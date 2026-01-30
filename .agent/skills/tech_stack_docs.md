---
skill_id: tech_documentation_ref
type: reference_library
topics: ["scikit-learn", "streamlit", "docker"]
---

# Reference: Technology Best Practices

## 1. Scikit-Learn Best Practices
* **Pipeline:** Siempre encapsular preprocesamiento.
    * *Correcto:* `Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])`
    * *Incorrecto:* Hacer `scaler.fit_transform()` fuera del pipeline.
* **Handling Imbalance:** Usar `class_weight='balanced'` en árboles de decisión es preferible a usar SMOTE en pipelines simples para producción.

## 2. Streamlit Patterns
* **Recarga:** Streamlit recarga todo el script en cada interacción.
* **Cache:** Usa `@st.cache_data` para cargar datos CSV y `@st.cache_resource` para cargar el modelo `.joblib`. Esto mejora el rendimiento drásticamente.

## 3. Estructura de Directorios Esperada
El sistema debe cumplir estrictamente:
/project_root
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app.py           (Frontend)
├── src/
│   ├── train_pipeline.py
│   ├── inference_pipeline.py
│   └── utils.py
├── data/
│   └── raw_data.csv
└── models/
    └── .gitkeep