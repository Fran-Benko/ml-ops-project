FROM python:3.10-slim

WORKDIR /app

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src ./src
COPY data ./data

# Crear directorios para outputs
RUN mkdir -p models models/drift_reports

# Configurar PYTHONPATH para imports relativos
ENV PYTHONPATH=/app

# Comando por defecto
CMD ["python", "src/train_pipeline_temporal.py", "--help"]