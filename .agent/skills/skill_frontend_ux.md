---
skill_id: ui_dashboard_developer
agent_role: Frontend Agent
trigger_events: ["metrics_updated", "user_interaction_design"]
tools_available: ["streamlit", "python_json", "plotting_lib"]
output_artifacts: ["app.py"]
dependencies: ["tech_streamlit_patterns"]
---

# Skill: Dashboard & User Interface

## Contexto Operativo
Eres la cara visible del sistema. Usarás **Streamlit** para interactuar con el backend de ML. No procesas datos pesados, solo invocas scripts y visualizas resultados.

## Instrucciones de Ejecución (`app.py`)

### Tarea A: Gestión de Estado (Session State)
Debes implementar lógica para mantener el estado entre recargas:
* Usa `st.session_state` para guardar si el modelo está entrenado o no.

### Tarea B: Diseño de Interfaz
El layout debe tener:
1.  **Sidebar:** Configuración y Botones de Control.
    * Botón `Entrenar Modelo`: Ejecuta `subprocess.run(["python", "src/train_pipeline.py"])`.
    * Botón `Predecir`: Ejecuta `subprocess.run(["python", "src/inference_pipeline.py"])`.
2.  **Main Area:**
    * **KPIs:** Lee `metrics.json` y muestra métricas grandes.
    * **Alertas:** Si `f1_score < 0.7`, muestra `st.error("Model Drift Detected")`.

### Tarea C: Simulación de Datos
Agrega una funcionalidad que permita al usuario "inyectar" datos falsos al CSV de entrada para probar el pipeline de inferencia en tiempo real.