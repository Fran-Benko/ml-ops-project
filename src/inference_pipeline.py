import pandas as pd
import joblib
import os
import argparse
import sys

MODEL_PATH = 'models/churn_pipeline.joblib'
OUTPUT_FILE = 'data/predictions.csv'

def load_inference_data(data_path):
    """
    Simula inferencia tomando un subset aleatorio de un dataset específico.
    Siempre varía la semilla para obtener resultados diferentes.
    """
    try:
        if not os.path.exists(data_path):
             print(f"Dataset no encontrado: {data_path}")
             return None, None

        df = pd.read_csv(data_path)
        
        # Tomar una muestra aleatoria de tamaño 10-50 para que varíe
        sample_size = min(len(df), random.randint(15, 30))
        sample = df.sample(sample_size).copy()
        ids = sample['EmployeeNumber'].values
        
        # --- Inyectar variabilidad 'Live' ---
        # Para que no sea EXACTAMENTE el mismo dato que el de entrenamiento, 
        # sumamos un pequeño ruido aleatorio a columnas numéricas clave
        numeric_cols = ['DailyRate', 'MonthlyIncome', 'DistanceFromHome', 'Age']
        for col in numeric_cols:
            if col in sample.columns:
                noise = np.random.normal(0, sample[col].std() * 0.05, size=len(sample))
                sample[col] = (sample[col] + noise).astype(int)
        
        # Eliminar columnas no deseadas si existen
        drop_cols = ['Attrition', 'EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
        X_new = sample.drop(columns=[c for c in drop_cols if c in sample.columns], errors='ignore')
        
        return ids, X_new
    except Exception as e:
        print(f"Error cargando datos de inferencia: {e}")
        return None, None

def run_inference(data_path="data/WA_Fn-UseC_-HR-Employee-Attrition.csv"):
    # 1. Cargar Modelo
    # Intentar buscar un modelo específico para este dataset o usar el genérico?
    # Por simplicidad del MVP, usamos el último modelo entrenado (el archivo 'churn_pipeline.joblib')
    # que se sobrescribe/actualiza o el que apunte el sistema. 
    # Idealmente, deberíamos pasar el modelo como argumento también.
    
    if not os.path.exists(MODEL_PATH):
        print("Error: No se encuentra el modelo entrenado (.joblib).")
        return

    print("Cargando modelo...")
    pipeline = joblib.load(MODEL_PATH)
    
    # 2. Cargar Nuevos Datos
    print(f"Usando fuente de datos: {data_path}")
    ids, X_new = load_inference_data(data_path)
    if X_new is None: return

    # 3. Validación de Schema
    required_cols = ['Age', 'DailyRate', 'Department'] 
    missing = [col for col in required_cols if col not in X_new.columns]
    if missing:
        print(f"Error de Schema: Faltan columnas {missing}")
        return

    try:
        # 4. Predicción
        print("Generando predicciones...")
        preds_class = pipeline.predict(X_new)
        preds_proba = pipeline.predict_proba(X_new)[:, 1]
        
        # 5. Output
        results = pd.DataFrame({
            'EmployeeID': ids,
            'Churn_Prediction': preds_class,
            'Churn_Probability': preds_proba
        })
        
        # Guardar con un nombre único o sobrescribir el último? 
        # Sobrescribimos 'predictions.csv' para que la UI lo muestre fácil,
        # pero podríamos guardar historial.
        results.to_csv(OUTPUT_FILE, index=False)
        print(f"Inferencia exitosa. Resultados guardados en '{OUTPUT_FILE}'.")
        print(results.head())
        
    except Exception as e:
        print(f"Error durante la predicción: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/WA_Fn-UseC_-HR-Employee-Attrition.csv", help="Path al dataset de entrada")
    args = parser.parse_args()
    
    run_inference(args.data)
