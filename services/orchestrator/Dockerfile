FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expone el puerto por defecto de Prefect
EXPOSE 4200

# Comando para iniciar el servidor de Prefect
CMD ["prefect", "server", "start", "--host", "0.0.0.0"]
