import pandas as pd
import json
import joblib
import argparse
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from src.utils import get_artifact_name, safe_path

# Configuración (Defaults)
DATA_PATH_DEFAULT = 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
METRICS_LATEST = 'models/latest_metrics.json'

def load_data(filepath):
    """Carga datos y realiza limpieza inicial."""
    try:
        df = pd.read_csv(filepath)
        print(f"Datos cargados: {df.shape}")
        
        # 1. Eliminar columnas irrelevantes o constantes
        # EmployeeNumber es ID, el resto son constantes según análisis
        drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # 2. Convertir target a binario
        if 'Attrition' in df.columns:
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
            
        return df
    except FileNotFoundError:
        print("Error: Archivo no encontrado.")
        raise

def get_pipeline(X_train):
    """Construye el pipeline de preprocesamiento y modelo."""
    
    # Identificar columnas automáticamente
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocesamiento para numéricas:
    # - Imputación por mediana (robusta a outliers)
    # - Escalado Standard (necesario para algunos modelos, buena práctica general)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocesamiento para categóricas:
    # - Imputación por moda
    # - OneHotEncoding. 'handle_unknown=ignore' es CRÍTICO para producción 
    #   si llegan categorías nuevas no vistas en train.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Unir transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Pipeline final con Random Forest
    # class_weight='balanced' maneja el desbalance de clases penalizando errores en la clase minoritaria
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    return pipeline

def train(dataset_path):
    # Validar ruta de entrada por seguridad
    try:
        validated_path = safe_path('data', os.path.basename(dataset_path))
    except Exception as e:
        print(f"Error de Seguridad: {e}")
        sys.exit(1)

    # 1. Carga
    print(f"Iniciando entrenamiento con dataset: {validated_path}")
    df = load_data(validated_path)
    
    # 2. Split
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    # Stratify asegura que la proporción de Churn (Yes/No) sea igual en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Construcción del Pipeline
    pipeline = get_pipeline(X_train)
    
    # 4. Optimización (GridSearchCV)
    # Buscamos reducir el overfitting y mejorar el recall (detectar fugas)
    param_grid = {
        'classifier__n_estimators': [100],
        'classifier__max_depth': [10, None],
    }
    
    print("Iniciando búsqueda de hiperparámetros...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Mejores parámetros: {grid_search.best_params_}")
    
    # 5. Evaluación
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'dataset_version': os.path.basename(validated_path),
        'timestamp': datetime.now().isoformat(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    print(f"ROC-AUC Test: {metrics['roc_auc']:.4f}")

    # --- Data Governance: Naming & Lineage ---
    model_filename = get_artifact_name(validated_path, 'model', '.joblib')
    metrics_filename = get_artifact_name(validated_path, 'metrics', '.json')
    
    model_path = os.path.join('models', model_filename)
    metrics_path = os.path.join('models', metrics_filename)
    
    # Guardar métricas
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 6. Serialización
    # Guardamos el pipeline ENTERO (preprocesamiento + modelo)
    joblib.dump(best_model, model_path)
    
    # Actualizar link simbólico o archivo 'latest' para la UI (opcional para UX)
    with open('models/latest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # --- MLflow Tracking ---
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Employee-Attrition-Refactored")
    
    with mlflow.start_run(run_name=f"Train_{metrics['dataset_version']}"):
        # Log Params
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("dataset", metrics['dataset_version'])
        
        # Log Metrics
        mlflow.log_metric("roc_auc", metrics['roc_auc'])
        mlflow.log_metric("f1_score_churn", metrics['classification_report']['1']['f1-score'])
        
        # Log Model
        mlflow.sklearn.log_model(best_model, "churn_pipeline")
        
        # Actualizar latest local para compatibilidad UI
        joblib.dump(best_model, 'models/churn_pipeline.joblib')
        
    print(f"✅ Entrenamiento completado y registrado en MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento Versionado")
    parser.add_argument("--data", type=str, required=True, help="Nombre del archivo CSV en la carpeta data/")
    args = parser.parse_args()
    
    train(args.data)