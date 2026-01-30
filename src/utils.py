import os
import re
from datetime import datetime

def get_next_dataset_version(directory='data', prefix='data_v', extension='.csv'):
    """
    Escanea el directorio y devuelve el siguiente nombre de archivo siguiendo la versión (ex: data_v1.csv -> data_v2.csv).
    Si no hay archivos, empieza en v1.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    
    if not files:
        return f"{prefix}1{extension}"
    
    # Extraer números de versión usando regex
    versions = []
    for f in files:
        match = re.search(rf'{prefix}(\d+){extension}', f)
        if match:
            versions.append(int(match.group(1)))
    
    next_version = max(versions) + 1 if versions else 1
    return f"{prefix}{next_version}{extension}"

def safe_path(base_dir, filename):
    """
    Previene Path Traversal validando que el archivo resida en el directorio base.
    """
    # Normalizar rutas
    absolute_base = os.path.abspath(base_dir)
    absolute_file = os.path.abspath(os.path.join(base_dir, filename))
    
    if not absolute_file.startswith(absolute_base):
        raise ValueError(f"Intento de Path Traversal detectado: {filename}")
        
    return absolute_file

def get_artifact_name(dataset_path, artifact_type='model', extension='.joblib'):
    """
    Genera un nombre de artefacto con timestamp y versión del dataset para trazabilidad.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(dataset_path).replace('.csv', '')
    return f"{artifact_type}_{timestamp}_{dataset_name}{extension}"
