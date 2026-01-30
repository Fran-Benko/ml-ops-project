---
skill_id: infra_container_expert
agent_role: DevOps Agent
trigger_events: ["code_generated", "deployment_request", "environment_setup"]
tools_available: ["docker", "docker-compose", "bash"]
output_artifacts: ["Dockerfile", "docker-compose.yml", "requirements.txt", "run.sh"]
dependencies: ["tech_docker_best_practices"]
---

# Skill: Infrastructure & Containerization

## Contexto Operativo
Tu misión es "dockerizar" el código generado por el ML Engineer y el UI Agent. Debes garantizar que el entorno sea reproducible y ligero.

## Instrucciones de Ejecución

### Tarea A: Definición de Dependencias (`requirements.txt`)
Analiza el código Python generado y crea el archivo de dependencias.
* **Must Have:** `scikit-learn`, `pandas`, `joblib`, `streamlit`.
* **Version Pinning:** Intenta fijar versiones mayores (ej. `pandas>=2.0`) para estabilidad.

### Tarea B: Construcción del Contenedor (`Dockerfile`)
Genera un Dockerfile optimizado siguiendo este patrón:
1.  **Base Image:** Usa `python:3.9-slim` (evita imágenes pesadas).
2.  **Working Dir:** `/app`.
3.  **Install:** Copia `requirements.txt` primero e instala, luego copia el código fuente (aprovecha el caché de capas de Docker).
4.  **Entrypoint:** El contenedor debe levantar por defecto la UI: `CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]`.

### Tarea C: Orquestación (`docker-compose.yml`)
Configura el servicio para montar volúmenes locales, permitiendo persistencia de modelos y logs:
```yaml
services:
  mlops-demo:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models  # Persistencia del modelo
      - ./data:/app/data      # Persistencia de datos