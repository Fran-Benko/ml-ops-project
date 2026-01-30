"""
Script de Comparación: Generador Original vs Temporal
=====================================================

Compara el rendimiento de modelos entrenados con:
1. Generador original (sintetic_gen.py) - Batches independientes
2. Generador temporal (temporal_generator.py) - Continuidad de cohortes

Métricas evaluadas:
- Performance del modelo (ROC-AUC, F1, Precision, Recall)
- Drift detection (PSI, KS-test)
- Validación temporal vs random
- Realismo de datos sintéticos

Autor: IBM Bob (Data Science Expert)
Fecha: 2026-01-20
Actualizado: 2026-01-21 - Integración con DataTypeNormalizer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings

from src.sintetic_gen import generate_hr_drift_dataset
from src.temporal_generator import TemporalHRGenerator
from src.drift_monitor import DriftMonitor
from src.temporal_validation import TemporalValidator
from src.data_type_normalizer import DataTypeNormalizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


class GeneratorComparator:
    """Compara generadores de datos sintéticos."""
    
    def __init__(self, seed_data_path: str = 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'):
        """
        Inicializa el comparador.
        
        Args:
            seed_data_path: Ruta a datos reales de referencia
        """
        self.seed_data = pd.read_csv(seed_data_path)
        print(f"✅ Datos semilla cargados: {self.seed_data.shape}")
        
        self.results = {
            'original_generator': {},
            'temporal_generator': {},
            'comparison': {}
        }
    
    def generate_data_original(self, n_per_batch: int = 400) -> pd.DataFrame:
        """Genera datos con generador original."""
        print("\n" + "="*70)
        print("GENERADOR ORIGINAL (Batches Independientes)")
        print("="*70)
        
        df = generate_hr_drift_dataset(n_per_batch=n_per_batch)
        
        # Agregar columna de mes simulada
        df['DataMonth'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(
            (df['Batch_ID'] - 1) * 30, unit='D'
        )
        
        print(f"✅ Datos generados: {df.shape}")
        print(f"   Batches: {df['Batch_ID'].nunique()}")
        
        return df
    
    def generate_data_temporal(self, n_months: int = 6, 
                              retention_rate: float = 0.85) -> pd.DataFrame:
        """Genera datos con generador temporal."""
        print("\n" + "="*70)
        print("GENERADOR TEMPORAL (Continuidad de Cohortes)")
        print("="*70)
        
        generator = TemporalHRGenerator(self.seed_data, start_date="2024-01-01")
        
        # Generar con drift gradual
        drift_schedule = [
            None,
            {'salary_increase': 0.02},
            {'salary_increase': 0.02, 'satisfaction_decay': -0.1},
            {'satisfaction_decay': -0.15, 'overtime_increase': 0.1},
            {'overtime_increase': 0.15},
            {'satisfaction_decay': -0.20},
        ]
        
        df = generator.generate_temporal_sequence(
            n_months=n_months,
            retention_rate=retention_rate,
            drift_schedule=drift_schedule
        )
        
        return df
    
    def build_model_pipeline(self, X_train: pd.DataFrame) -> Pipeline:
        """Construye pipeline de modelo."""
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        return pipeline
    
    def evaluate_with_random_split(self, data: pd.DataFrame,
                                   generator_name: str) -> dict:
        """Evalúa modelo con split aleatorio (baseline)."""
        print(f"\n📊 Evaluando {generator_name} con Random Split...")
        
        # Preparar datos
        X = data.drop(columns=['Attrition', 'DataMonth'], errors='ignore')
        if 'Batch_ID' in X.columns:
            X = X.drop(columns=['Batch_ID'])
        
        # Convertir columnas categóricas a string para evitar errores de encoding
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].astype(str)
        
        y = data['Attrition']
        if y.dtype == 'object':
            y = y.map({'Yes': 1, 'No': 0})
        
        # Split aleatorio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Entrenar
        model = self.build_model_pipeline(X_train)
        model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1_score': f1_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics, model
    
    def evaluate_with_temporal_validation(self, data: pd.DataFrame,
                                         generator_name: str,
                                         n_splits: int = 4) -> dict:
        """Evalúa modelo con validación temporal."""
        print(f"\n📊 Evaluando {generator_name} con Temporal Validation...")
        
        # Preparar datos
        X = data.drop(columns=['Attrition', 'DataMonth'], errors='ignore')
        if 'Batch_ID' in X.columns:
            X = X.drop(columns=['Batch_ID'])
        
        # Crear modelo base
        model = self.build_model_pipeline(X)
        
        # Validación temporal
        validator = TemporalValidator(date_column='DataMonth', target_column='Attrition')
        
        results = validator.walk_forward_validation(
            data, model, n_splits=n_splits, strategy='expanding'
        )
        
        return results
    
    def analyze_drift(self, data: pd.DataFrame, generator_name: str) -> dict:
        """Analiza drift en datos generados."""
        print(f"\n🔍 Analizando Drift en {generator_name}...")
        
        # Usar primer período como referencia
        periods = sorted(data['DataMonth'].unique())
        reference_data = data[data['DataMonth'] == periods[0]]
        new_data = data[data['DataMonth'] == periods[-1]]
        
        # Inicializar monitor con normalización automática
        # El DriftMonitor ahora maneja la normalización de tipos internamente
        monitor = DriftMonitor(reference_data, target_col='Attrition',
                              auto_normalize=True, verbose=False)
        
        # Detectar drift (los datos se normalizan automáticamente)
        drift_report = monitor.generate_drift_report(new_data)
        
        return drift_report
    
    def analyze_data_realism(self, data: pd.DataFrame, generator_name: str) -> dict:
        """Analiza realismo de datos sintéticos."""
        print(f"\n🔬 Analizando Realismo de {generator_name}...")
        
        realism_metrics = {}
        
        # 1. Continuidad de empleados (solo para temporal)
        if 'EmployeeNumber' in data.columns:
            unique_employees = data['EmployeeNumber'].nunique()
            total_records = len(data)
            periods = data['DataMonth'].nunique()
            
            expected_if_independent = total_records  # Todos diferentes
            expected_if_continuous = total_records / periods  # Mismos empleados
            
            continuity_score = 1 - (unique_employees - expected_if_continuous) / (
                expected_if_independent - expected_if_continuous
            )
            
            realism_metrics['employee_continuity'] = {
                'unique_employees': unique_employees,
                'total_records': total_records,
                'continuity_score': max(0, min(1, continuity_score)),
                'interpretation': 'HIGH' if continuity_score > 0.7 else 'LOW'
            }
            
            print(f"   Empleados únicos: {unique_employees}")
            print(f"   Continuidad: {continuity_score:.2%}")
        
        # 2. Tasa de attrition por período
        attrition_by_period = data.groupby('DataMonth')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() if x.dtype == 'object' else x.mean()
        )
        
        realism_metrics['attrition_stability'] = {
            'mean': attrition_by_period.mean(),
            'std': attrition_by_period.std(),
            'cv': attrition_by_period.std() / attrition_by_period.mean(),  # Coef. variación
            'interpretation': 'STABLE' if attrition_by_period.std() < 0.05 else 'VOLATILE'
        }
        
        print(f"   Attrition promedio: {attrition_by_period.mean():.2%}")
        print(f"   Volatilidad: {attrition_by_period.std():.4f}")
        
        # 3. Distribución de edad (debe ser realista)
        age_stats = {
            'mean': data['Age'].mean(),
            'std': data['Age'].std(),
            'min': data['Age'].min(),
            'max': data['Age'].max()
        }
        
        # Comparar con datos reales
        real_age_mean = self.seed_data['Age'].mean()
        age_deviation = abs(age_stats['mean'] - real_age_mean) / real_age_mean
        
        realism_metrics['age_distribution'] = {
            **age_stats,
            'deviation_from_real': age_deviation,
            'interpretation': 'REALISTIC' if age_deviation < 0.1 else 'UNREALISTIC'
        }
        
        print(f"   Edad promedio: {age_stats['mean']:.1f} años")
        print(f"   Desviación vs real: {age_deviation:.2%}")
        
        return realism_metrics
    
    def run_full_comparison(self) -> dict:
        """Ejecuta comparación completa."""
        print("\n" + "="*70)
        print("COMPARACIÓN COMPLETA: GENERADOR ORIGINAL VS TEMPORAL")
        print("="*70)
        
        # 1. Generar datos
        data_original = self.generate_data_original(n_per_batch=400)
        data_temporal = self.generate_data_temporal(n_months=6, retention_rate=0.85)
        
        # 2. Evaluar con random split
        metrics_orig_random, model_orig = self.evaluate_with_random_split(
            data_original, "Original"
        )
        metrics_temp_random, model_temp = self.evaluate_with_random_split(
            data_temporal, "Temporal"
        )
        
        self.results['original_generator']['random_split'] = metrics_orig_random
        self.results['temporal_generator']['random_split'] = metrics_temp_random
        
        # 3. Evaluar con validación temporal
        # Ajustar n_splits según períodos disponibles
        n_periods_orig = data_original['DataMonth'].nunique()
        n_periods_temp = data_temporal['DataMonth'].nunique()
        
        n_splits_orig = min(2, n_periods_orig - 1)  # Original tiene 3 períodos
        n_splits_temp = min(4, n_periods_temp - 1)  # Temporal tiene 7 períodos
        
        metrics_orig_temporal = self.evaluate_with_temporal_validation(
            data_original, "Original", n_splits=n_splits_orig
        )
        metrics_temp_temporal = self.evaluate_with_temporal_validation(
            data_temporal, "Temporal", n_splits=n_splits_temp
        )
        
        self.results['original_generator']['temporal_validation'] = metrics_orig_temporal
        self.results['temporal_generator']['temporal_validation'] = metrics_temp_temporal
        
        # 4. Analizar drift
        drift_orig = self.analyze_drift(data_original, "Original")
        drift_temp = self.analyze_drift(data_temporal, "Temporal")
        
        self.results['original_generator']['drift_analysis'] = drift_orig
        self.results['temporal_generator']['drift_analysis'] = drift_temp
        
        # 5. Analizar realismo
        realism_orig = self.analyze_data_realism(data_original, "Original")
        realism_temp = self.analyze_data_realism(data_temporal, "Temporal")
        
        self.results['original_generator']['realism'] = realism_orig
        self.results['temporal_generator']['realism'] = realism_temp
        
        # 6. Comparación final
        self._generate_comparison_summary()
        
        return self.results
    
    def _generate_comparison_summary(self):
        """Genera resumen de comparación."""
        print("\n" + "="*70)
        print("RESUMEN DE COMPARACIÓN")
        print("="*70)
        
        # Performance
        print("\n📊 PERFORMANCE (Random Split):")
        orig_auc = self.results['original_generator']['random_split']['roc_auc']
        temp_auc = self.results['temporal_generator']['random_split']['roc_auc']
        print(f"   Original: ROC-AUC = {orig_auc:.4f}")
        print(f"   Temporal: ROC-AUC = {temp_auc:.4f}")
        print(f"   Diferencia: {(temp_auc - orig_auc):.4f}")
        
        # Temporal validation
        print("\n📊 PERFORMANCE (Temporal Validation):")
        orig_temp_auc = self.results['original_generator']['temporal_validation']['mean_metrics']['roc_auc']
        temp_temp_auc = self.results['temporal_generator']['temporal_validation']['mean_metrics']['roc_auc']
        print(f"   Original: ROC-AUC = {orig_temp_auc:.4f}")
        print(f"   Temporal: ROC-AUC = {temp_temp_auc:.4f}")
        print(f"   Diferencia: {(temp_temp_auc - orig_temp_auc):.4f}")
        
        # Data leakage detection
        orig_leakage = abs(orig_auc - orig_temp_auc)
        temp_leakage = abs(temp_auc - temp_temp_auc)
        print("\n⚠️ DATA LEAKAGE DETECTION:")
        print(f"   Original: {orig_leakage:.4f} {'🚨 ALTO' if orig_leakage > 0.05 else '✅ BAJO'}")
        print(f"   Temporal: {temp_leakage:.4f} {'🚨 ALTO' if temp_leakage > 0.05 else '✅ BAJO'}")
        
        # Realismo
        print("\n🔬 REALISMO DE DATOS:")
        if 'employee_continuity' in self.results['temporal_generator']['realism']:
            temp_continuity = self.results['temporal_generator']['realism']['employee_continuity']['continuity_score']
            print(f"   Temporal - Continuidad de empleados: {temp_continuity:.2%}")
        
        orig_attrition_cv = self.results['original_generator']['realism']['attrition_stability']['cv']
        temp_attrition_cv = self.results['temporal_generator']['realism']['attrition_stability']['cv']
        print(f"   Original - Volatilidad attrition: {orig_attrition_cv:.4f}")
        print(f"   Temporal - Volatilidad attrition: {temp_attrition_cv:.4f}")
        
        # Drift
        print("\n🔍 DRIFT DETECTION:")
        orig_drift_alerts = self.results['original_generator']['drift_analysis']['summary']['covariate_alerts']
        temp_drift_alerts = self.results['temporal_generator']['drift_analysis']['summary']['covariate_alerts']
        print(f"   Original: {orig_drift_alerts} alertas")
        print(f"   Temporal: {temp_drift_alerts} alertas")
        
        # Veredicto final
        print("\n" + "="*70)
        print("VEREDICTO FINAL")
        print("="*70)
        
        score_orig = 0
        score_temp = 0
        
        # Criterio 1: Menor data leakage
        if temp_leakage < orig_leakage:
            score_temp += 2
            print("✅ Temporal gana en: Menor data leakage")
        else:
            score_orig += 2
            print("✅ Original gana en: Menor data leakage")
        
        # Criterio 2: Realismo (continuidad)
        if 'employee_continuity' in self.results['temporal_generator']['realism']:
            score_temp += 2
            print("✅ Temporal gana en: Continuidad de empleados")
        
        # Criterio 3: Estabilidad de attrition
        if temp_attrition_cv < orig_attrition_cv:
            score_temp += 1
            print("✅ Temporal gana en: Estabilidad de attrition")
        else:
            score_orig += 1
            print("✅ Original gana en: Estabilidad de attrition")
        
        # Criterio 4: Performance temporal
        if temp_temp_auc > orig_temp_auc:
            score_temp += 2
            print("✅ Temporal gana en: Performance con validación temporal")
        else:
            score_orig += 2
            print("✅ Original gana en: Performance con validación temporal")
        
        print(f"\n🏆 SCORE FINAL:")
        print(f"   Original: {score_orig} puntos")
        print(f"   Temporal: {score_temp} puntos")
        
        winner = "TEMPORAL" if score_temp > score_orig else "ORIGINAL"
        print(f"\n🎯 GANADOR: {winner}")
        
        self.results['comparison']['winner'] = winner
        self.results['comparison']['scores'] = {'original': score_orig, 'temporal': score_temp}
    
    def save_results(self, output_path: str = 'experiments/comparison_results.json'):
        """Guarda resultados en JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convertir a formato serializable
        results_serializable = json.loads(
            json.dumps(self.results, default=str)
        )
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n💾 Resultados guardados en: {output_path}")


if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENTO: COMPARACIÓN DE GENERADORES")
    print("="*70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ejecutar comparación
    comparator = GeneratorComparator()
    results = comparator.run_full_comparison()
    
    # Guardar resultados
    comparator.save_results()
    
    print("\n✅ Experimento completado")

# Made with Bob
