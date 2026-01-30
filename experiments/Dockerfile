FROM python:3.10-slim

WORKDIR /app

# Copiar requirements
COPY ../requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY ../src ./src
COPY ../data ./data
COPY ../experiments ./experiments

# Crear directorios necesarios
RUN mkdir -p models experiments/results

# Comando por defecto
CMD ["python", "experiments/compare_generators.py"]