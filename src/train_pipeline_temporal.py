"""
Pipeline de Entrenamiento con Validación Temporal y Drift Monitoring
====================================================================

Integra:
- TemporalHRGenerator: Generación de datos con continuidad temporal
- DriftMonitor: Detección avanzada de drift (PSI, KS-test, Wasserstein)
- TemporalValidator: Walk-forward validation
- DataTypeNormalizer: Normalización automática de tipos
- MLflow tracking
- Data governance

Autor: Franco Benko
Fecha: 2026-01-20
Actualizado: 2026-01-21 - Integración con DataTypeNormalizer
"""

import pandas as pd
import json
import joblib
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Imports locales
from src.utils import get_artifact_name, safe_path
from src.temporal_generator import TemporalHRGenerator
from src.drift_monitor import DriftMonitor
from src.temporal_validation import TemporalValidator
from src.data_type_normalizer import DataTypeNormalizer

# Configuración
DATA_PATH_DEFAULT = 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
METRICS_LATEST = 'models/latest_metrics.json'
DRIFT_REPORTS_DIR = 'models/drift_reports'


def load_data(filepath):
    """Carga datos y realiza limpieza inicial."""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Datos cargados: {df.shape}")
        
        # Eliminar columnas irrelevantes o constantes
        drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # Convertir target a binario
        if 'Attrition' in df.columns:
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
            
        return df
    except FileNotFoundError:
        print("❌ Error: Archivo no encontrado.")
        raise


def get_pipeline(X_train):
    """Construye el pipeline de preprocesamiento y modelo."""
    
    # Identificar columnas automáticamente
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocesamiento para numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocesamiento para categóricas
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
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    return pipeline


def generate_temporal_data(seed_data, n_months=6, scenario='baseline'):
    """
    Genera datos sintéticos con continuidad temporal.
    
    Args:
        seed_data: DataFrame con datos iniciales
        n_months: Número de meses a generar
        scenario: Escenario de drift ('baseline', 'economic_recession', etc.)
    
    Returns:
        DataFrame con datos temporales generados
    """
    print(f"\n{'='*70}")
    print(f"🎲 GENERACIÓN DE DATOS TEMPORALES")
    print(f"{'='*70}")
    print(f"Escenario: {scenario}")
    print(f"Meses a generar: {n_months}")
    
    generator = TemporalHRGenerator(seed_data)
    
    # Crear schedule de escenarios (todos los meses con el mismo escenario)
    scenario_schedule = [scenario] * n_months
    
    temporal_data = generator.generate_temporal_sequence(
        n_months=n_months,
        scenario_schedule=scenario_schedule
    )
    
    print(f"✅ Generados {len(temporal_data)} registros en {n_months + 1} períodos")
    return temporal_data


def monitor_drift(reference_data, current_data, output_dir='models/drift_reports'):
    """
    Monitorea drift entre datos de referencia y actuales.
    
    Args:
        reference_data: DataFrame de referencia
        current_data: DataFrame actual
        output_dir: Directorio para guardar reportes
    
    Returns:
        dict con resultados del análisis de drift
    """
    print(f"\n{'='*70}")
    print(f"🔍 ANÁLISIS DE DRIFT")
    print(f"{'='*70}")
    
    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Eliminar columnas temporales si existen
    ref_clean = reference_data.drop(columns=['period', 'EmployeeNumber'], errors='ignore')
    curr_clean = current_data.drop(columns=['period', 'EmployeeNumber'], errors='ignore')
    
    # Inicializar monitor con normalización automática
    # El DriftMonitor ahora maneja la normalización de tipos internamente
    monitor = DriftMonitor(ref_clean, target_col='Attrition', auto_normalize=True, verbose=True)
    
    # Detectar drift (los datos se normalizan automáticamente)
    drift_report = monitor.generate_drift_report(curr_clean)
    
    # Guardar reporte
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f'drift_report_{timestamp}.json')
    
    with open(report_path, 'w') as f:
        json.dump(drift_report, f, indent=2)
    
    print(f"💾 Reporte guardado en: {report_path}")
    
    # Mostrar resumen
    summary = drift_report['summary']
    print(f"\n📊 RESUMEN:")
    print(f"   Features monitoreadas: {summary['total_features_monitored']}")
    print(f"   Alertas de covariate shift: {summary['covariate_alerts']}")
    print(f"   Concept drift detectado: {'SÍ' if summary['concept_drift_detected'] else 'NO'}")
    print(f"   Estado general: {summary['overall_status']}")
    
    return drift_report


def temporal_validation(data, n_splits=3, strategy='expanding'):
    """
    Realiza validación temporal walk-forward.
    
    Args:
        data: DataFrame con columna 'period'
        n_splits: Número de splits temporales
        strategy: 'expanding' o 'rolling'
    
    Returns:
        dict con resultados de validación
    """
    print(f"\n{'='*70}")
    print(f"📊 VALIDACIÓN TEMPORAL")
    print(f"{'='*70}")
    
    # Separar features y target
    X = data.drop(columns=['Attrition'])
    y = data['Attrition']
    
    # Crear validador
    validator = TemporalValidator(strategy=strategy, n_splits=n_splits)
    
    # Crear pipeline
    pipeline = get_pipeline(X)
    
    # Ejecutar validación
    results = validator.validate(pipeline, X, y)
    
    # Mostrar resumen
    print(f"\n📊 RESUMEN:")
    print(f"   ROC-AUC: {results['mean_metrics']['roc_auc']:.4f} ± {results['std_metrics']['roc_auc']:.4f}")
    print(f"   F1-Score: {results['mean_metrics']['f1_score']:.4f} ± {results['std_metrics']['f1_score']:.4f}")
    print(f"   Precision: {results['mean_metrics']['precision']:.4f} ± {results['std_metrics']['precision']:.4f}")
    print(f"   Recall: {results['mean_metrics']['recall']:.4f} ± {results['std_metrics']['recall']:.4f}")
    
    return results


def train_with_temporal_validation(
    dataset_path,
    use_temporal_generation=False,
    n_temporal_months=6,
    drift_scenario='baseline',
    use_temporal_validation=True,
    n_splits=3
):
    """
    Pipeline de entrenamiento con validación temporal y drift monitoring.
    
    Args:
        dataset_path: Ruta al dataset inicial
        use_temporal_generation: Si True, genera datos temporales sintéticos
        n_temporal_months: Número de meses a generar
        drift_scenario: Escenario de drift para generación
        use_temporal_validation: Si True, usa walk-forward validation
        n_splits: Número de splits para validación temporal
    """
    # Validar ruta de entrada
    try:
        validated_path = safe_path('data', os.path.basename(dataset_path))
    except Exception as e:
        print(f"❌ Error de Seguridad: {e}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"🚀 PIPELINE DE ENTRENAMIENTO TEMPORAL")
    print(f"{'='*70}")
    print(f"Dataset: {validated_path}")
    print(f"Generación temporal: {'SÍ' if use_temporal_generation else 'NO'}")
    print(f"Validación temporal: {'SÍ' if use_temporal_validation else 'NO'}")
    
    # 1. Carga de datos iniciales
    df_seed = load_data(validated_path)
    
    # 2. Generación temporal (opcional)
    if use_temporal_generation:
        df_temporal = generate_temporal_data(
            df_seed,
            n_months=n_temporal_months,
            scenario=drift_scenario
        )
        
        # TODO: Monitorear drift entre seed y temporal
        # Temporalmente deshabilitado por problemas de tipos mixtos
        # drift_report = monitor_drift(df_seed, df_temporal)
        drift_report = None
        print("\n⚠️ Drift monitoring temporalmente deshabilitado")
        
        # Usar datos temporales para entrenamiento
        df = df_temporal
    else:
        df = df_seed
        drift_report = None
    
    # 3. Validación temporal (opcional)
    if use_temporal_validation and 'period' in df.columns:
        validation_results = temporal_validation(df, n_splits=n_splits)
    else:
        validation_results = None
        print("\n⚠️ Validación temporal omitida (sin columna 'period' o deshabilitada)")
    
    # 4. Entrenamiento final con todos los datos
    print(f"\n{'='*70}")
    print(f"🎯 ENTRENAMIENTO FINAL")
    print(f"{'='*70}")
    
    X = df.drop(columns=['Attrition', 'period', 'EmployeeNumber'], errors='ignore')
    y = df['Attrition']
    
    # Asegurar que y sea numérico y sin NaN
    if y.dtype == 'object':
        y = y.map({'Yes': 1, 'No': 0})
    
    # Eliminar filas con NaN en target
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    y = y.astype(int)
    
    # Normalizar tipos de datos automáticamente
    normalizer = DataTypeNormalizer(verbose=False)
    X = normalizer.normalize_dataframe(X)
    
    # Split tradicional para evaluación final
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Construcción del Pipeline
    pipeline = get_pipeline(X_train)
    
    # Optimización (GridSearchCV)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
    }
    
    print("🔍 Iniciando búsqueda de hiperparámetros...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✅ Mejores parámetros: {grid_search.best_params_}")
    
    # 5. Evaluación
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'dataset_version': os.path.basename(validated_path),
        'timestamp': datetime.now().isoformat(),
        'temporal_generation': use_temporal_generation,
        'temporal_validation': use_temporal_validation,
        'n_temporal_months': n_temporal_months if use_temporal_generation else None,
        'drift_scenario': drift_scenario if use_temporal_generation else None,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'best_params': grid_search.best_params_,
        'validation_results': validation_results,
        'drift_summary': drift_report['summary'] if drift_report else None
    }
    
    print(f"\n📊 ROC-AUC Test: {metrics['roc_auc']:.4f}")
    
    # 6. Data Governance: Naming & Lineage
    model_filename = get_artifact_name(validated_path, 'model_temporal', '.joblib')
    metrics_filename = get_artifact_name(validated_path, 'metrics_temporal', '.json')
    
    model_path = os.path.join('models', model_filename)
    metrics_path = os.path.join('models', metrics_filename)
    
    # Guardar métricas
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Serialización del modelo
    joblib.dump(best_model, model_path)
    
    # Actualizar latest
    with open('models/latest_metrics_temporal.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 7. MLflow Tracking
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        import mlflow
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment("Employee-Attrition-Temporal")
        
        with mlflow.start_run(run_name=f"Temporal_{metrics['dataset_version']}"):
            # Log Params
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("dataset", metrics['dataset_version'])
            mlflow.log_param("temporal_generation", use_temporal_generation)
            mlflow.log_param("temporal_validation", use_temporal_validation)
            if use_temporal_generation:
                mlflow.log_param("n_temporal_months", n_temporal_months)
                mlflow.log_param("drift_scenario", drift_scenario)
            
            # Log Metrics
            mlflow.log_metric("roc_auc", metrics['roc_auc'])
            mlflow.log_metric("f1_score_churn", metrics['classification_report']['1']['f1-score'])
            
            if validation_results:
                mlflow.log_metric("temporal_roc_auc_mean", validation_results['mean_metrics']['roc_auc'])
                mlflow.log_metric("temporal_roc_auc_std", validation_results['std_metrics']['roc_auc'])
            
            if drift_report:
                mlflow.log_metric("drift_alerts", drift_report['summary']['covariate_alerts'])
            
            # Log Model
            mlflow.sklearn.log_model(best_model, "churn_pipeline_temporal")
            
            # Log artifacts
            mlflow.log_artifact(metrics_path)
            if drift_report:
                drift_report_path = os.path.join(
                    DRIFT_REPORTS_DIR,
                    f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                mlflow.log_artifact(drift_report_path)
        
        print(f"✅ Entrenamiento registrado en MLflow")
    except Exception as e:
        print(f"⚠️ MLflow tracking falló: {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE COMPLETADO")
    print(f"{'='*70}")
    print(f"Modelo guardado: {model_path}")
    print(f"Métricas guardadas: {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de Entrenamiento con Validación Temporal"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Nombre del archivo CSV en la carpeta data/"
    )
    parser.add_argument(
        "--temporal-gen",
        action="store_true",
        help="Generar datos temporales sintéticos"
    )
    parser.add_argument(
        "--n-months",
        type=int,
        default=6,
        help="Número de meses a generar (default: 6)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="baseline",
        choices=['baseline', 'economic_recession', 'tech_boom', 'high_competition'],
        help="Escenario de drift (default: baseline)"
    )
    parser.add_argument(
        "--temporal-val",
        action="store_true",
        help="Usar validación temporal walk-forward"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Número de splits para validación temporal (default: 3)"
    )
    
    args = parser.parse_args()
    
    train_with_temporal_validation(
        dataset_path=args.data,
        use_temporal_generation=args.temporal_gen,
        n_temporal_months=args.n_months,
        drift_scenario=args.scenario,
        use_temporal_validation=args.temporal_val,
        n_splits=args.n_splits
    )

# Made with Bob
