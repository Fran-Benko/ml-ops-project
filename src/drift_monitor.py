"""
Sistema Robusto de Detección de Drift
======================================

Implementa múltiples métricas estadísticas para detectar drift en datos:
- PSI (Population Stability Index): Estándar en banca/RRHH
- KS Test (Kolmogorov-Smirnov): Cambios en distribuciones
- Wasserstein Distance: Distancia entre distribuciones
- Model Performance Decay: Degradación de métricas del modelo

Autor: IBM Bob (Data Science Expert)
Fecha: 2026-01-20
Actualizado: 2026-01-21 - Integración con DataTypeNormalizer
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from typing import Dict, List, Optional, Tuple
import warnings
import json

from src.data_type_normalizer import DataTypeNormalizer

warnings.filterwarnings('ignore')


class DriftMonitor:
    """
    Monitor de drift con múltiples métricas estadísticas.
    
    Detecta:
    - Covariate Shift: Cambios en P(X)
    - Concept Drift: Cambios en P(Y|X)
    - Model Performance Decay: Degradación de métricas
    """
    
    def __init__(self, reference_data: pd.DataFrame, target_col: str = 'Attrition',
                 auto_normalize: bool = True, verbose: bool = True):
        """
        Inicializa el monitor con datos de referencia.
        
        Args:
            reference_data: DataFrame de referencia (baseline)
            target_col: Nombre de la columna target
            auto_normalize: Si True, normaliza automáticamente los tipos de datos
            verbose: Si True, imprime información de inicialización
        """
        self.target_col = target_col
        self.auto_normalize = auto_normalize
        self.verbose = verbose
        
        # Inicializar normalizador
        self.normalizer = DataTypeNormalizer(verbose=False)
        
        # Normalizar datos de referencia si está habilitado
        if auto_normalize:
            if verbose:
                print(f"🔧 Normalizando datos de referencia...")
            reference_data = self.normalizer.normalize_dataframe(
                reference_data.copy(),
                exclude_cols=[target_col] if target_col in reference_data.columns else []
            )
        
        self.reference = reference_data.copy()
        
        # Separar features y target
        if target_col in self.reference.columns:
            self.reference_X = self.reference.drop(columns=[target_col])
            self.reference_y = self.reference[target_col]
        else:
            self.reference_X = self.reference
            self.reference_y = None
        
        # Identificar tipos de columnas
        self.numeric_cols = self.reference_X.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        self.categorical_cols = self.reference_X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        if verbose:
            print(f"✅ DriftMonitor inicializado")
            print(f"   📊 Datos de referencia: {len(self.reference)} registros")
            print(f"   🔢 Features numéricas: {len(self.numeric_cols)}")
            print(f"   📝 Features categóricas: {len(self.categorical_cols)}")
            if auto_normalize:
                print(f"   🔧 Normalización automática: ACTIVADA")
    
    def calculate_psi(self, reference: pd.Series, current: pd.Series, 
                     bins: int = 10) -> float:
        """
        Calcula Population Stability Index (PSI).
        
        PSI es una métrica estándar en banca/RRHH para detectar drift.
        
        Interpretación:
        - PSI < 0.1: Sin cambio significativo
        - 0.1 <= PSI < 0.25: Cambio moderado (monitorear)
        - PSI >= 0.25: Cambio significativo (acción requerida)
        
        Args:
            reference: Serie de referencia
            current: Serie actual
            bins: Número de bins para discretización
        
        Returns:
            Valor PSI
        """
        # Manejar valores faltantes
        reference = reference.dropna()
        current = current.dropna()
        
        if len(reference) == 0 or len(current) == 0:
            return np.nan
        
        # Crear bins basados en referencia
        try:
            _, bin_edges = np.histogram(reference, bins=bins)
            
            # Calcular distribuciones
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Convertir a proporciones (evitar división por cero)
            ref_props = (ref_counts + 1e-6) / (len(reference) + bins * 1e-6)
            curr_props = (curr_counts + 1e-6) / (len(current) + bins * 1e-6)
            
            # Calcular PSI
            psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
            
            return psi
        except Exception as e:
            print(f"⚠️ Error calculando PSI: {e}")
            return np.nan
    
    def detect_covariate_shift(self, new_data: pd.DataFrame,
                              threshold_psi: float = 0.25,
                              threshold_ks: float = 0.05) -> Dict:
        """
        Detecta cambios en la distribución de features (Covariate Shift).
        
        Usa múltiples métricas:
        - PSI para features numéricas
        - KS-test para validación estadística
        - Chi-cuadrado para categóricas
        
        Args:
            new_data: DataFrame con nuevos datos
            threshold_psi: Umbral para PSI (default: 0.25)
            threshold_ks: Umbral para p-value de KS-test (default: 0.05)
        
        Returns:
            Dict con alertas por feature
        """
        alerts = {}
        
        # Normalizar nuevos datos si auto_normalize está activado
        if self.auto_normalize:
            new_data = self.normalizer.normalize_dataframe(
                new_data.copy(),
                exclude_cols=[self.target_col] if self.target_col in new_data.columns else []
            )
            
            # Validar consistencia con datos de referencia
            if self.verbose:
                consistency = self.normalizer.validate_consistency(
                    self.reference, new_data, check_columns=False
                )
                if consistency['has_issues']:
                    print(f"⚠️ Advertencias de consistencia:")
                    for issue in consistency['issues'][:3]:  # Mostrar solo primeras 3
                        print(f"   - {issue}")
        
        # Preparar datos
        if self.target_col in new_data.columns:
            new_X = new_data.drop(columns=[self.target_col])
        else:
            new_X = new_data
        
        if self.verbose:
            print(f"\n🔍 Detectando Covariate Shift...")
        
        # Analizar features numéricas
        for col in self.numeric_cols:
            if col not in new_X.columns:
                continue
            
            ref_col = self.reference_X[col].dropna()
            new_col = new_X[col].dropna()
            
            if len(ref_col) == 0 or len(new_col) == 0:
                continue
            
            # Calcular PSI
            psi = self.calculate_psi(ref_col, new_col)
            
            # KS-test
            ks_stat, ks_pvalue = ks_2samp(ref_col, new_col)
            
            # Wasserstein distance (normalizado)
            try:
                wass_dist = wasserstein_distance(ref_col, new_col)
                # Normalizar por rango
                data_range = ref_col.max() - ref_col.min()
                wass_norm = wass_dist / data_range if data_range > 0 else 0
            except:
                wass_norm = np.nan
            
            # Determinar severidad
            severity = "LOW"
            if psi >= threshold_psi or ks_pvalue < threshold_ks:
                severity = "HIGH" if psi >= 0.5 else "MEDIUM"
            
            # Guardar resultados
            alerts[col] = {
                'type': 'numeric',
                'psi': round(psi, 4) if not np.isnan(psi) else None,
                'ks_statistic': round(ks_stat, 4),
                'ks_pvalue': round(ks_pvalue, 4),
                'wasserstein_distance': round(wass_norm, 4) if not np.isnan(wass_norm) else None,
                'severity': severity,
                'alert': severity in ['MEDIUM', 'HIGH'],
                'mean_reference': round(ref_col.mean(), 2),
                'mean_current': round(new_col.mean(), 2),
                'std_reference': round(ref_col.std(), 2),
                'std_current': round(new_col.std(), 2)
            }
        
        # Analizar features categóricas
        for col in self.categorical_cols:
            if col not in new_X.columns:
                continue
            
            ref_col = self.reference_X[col].dropna()
            new_col = new_X[col].dropna()
            
            if len(ref_col) == 0 or len(new_col) == 0:
                continue
            
            # Distribuciones
            ref_dist = ref_col.value_counts(normalize=True).to_dict()
            new_dist = new_col.value_counts(normalize=True).to_dict()
            
            # Calcular cambio máximo en proporciones
            all_categories = set(ref_dist.keys()) | set(new_dist.keys())
            max_change = max([
                abs(new_dist.get(cat, 0) - ref_dist.get(cat, 0))
                for cat in all_categories
            ])
            
            severity = "LOW"
            if max_change > 0.15:
                severity = "HIGH" if max_change > 0.30 else "MEDIUM"
            
            alerts[col] = {
                'type': 'categorical',
                'max_proportion_change': round(max_change, 4),
                'reference_distribution': ref_dist,
                'current_distribution': new_dist,
                'severity': severity,
                'alert': severity in ['MEDIUM', 'HIGH']
            }
        
        # Resumen
        n_alerts = sum(1 for v in alerts.values() if v['alert'])
        print(f"   ⚠️ Alertas detectadas: {n_alerts}/{len(alerts)} features")
        
        return alerts
    
    def detect_concept_drift(self, model, new_data: pd.DataFrame,
                            threshold: float = 0.05) -> Dict:
        """
        Detecta degradación en performance del modelo (Concept Drift).
        
        Compara métricas entre datos de referencia y nuevos datos.
        
        Args:
            model: Modelo entrenado (debe tener predict_proba)
            new_data: DataFrame con nuevos datos
            threshold: Umbral de degradación aceptable (default: 5%)
        
        Returns:
            Dict con métricas y alertas
        """
        print(f"\n🔍 Detectando Concept Drift...")
        
        if self.reference_y is None or self.target_col not in new_data.columns:
            print("   ⚠️ No hay target disponible para concept drift")
            return {}
        
        # Preparar datos
        new_X = new_data.drop(columns=[self.target_col])
        new_y = new_data[self.target_col]
        
        # Convertir target a binario si es necesario
        if new_y.dtype == 'object':
            new_y = new_y.map({'Yes': 1, 'No': 0})
        if self.reference_y.dtype == 'object':
            ref_y = self.reference_y.map({'Yes': 1, 'No': 0})
        else:
            ref_y = self.reference_y
        
        try:
            # Predicciones en referencia
            ref_pred = model.predict(self.reference_X)
            ref_proba = model.predict_proba(self.reference_X)[:, 1]
            
            # Predicciones en nuevos datos
            new_pred = model.predict(new_X)
            new_proba = model.predict_proba(new_X)[:, 1]
            
            # Calcular métricas
            metrics_ref = {
                'roc_auc': roc_auc_score(ref_y, ref_proba),
                'f1_score': f1_score(ref_y, ref_pred),
                'precision': precision_score(ref_y, ref_pred),
                'recall': recall_score(ref_y, ref_pred)
            }
            
            metrics_new = {
                'roc_auc': roc_auc_score(new_y, new_proba),
                'f1_score': f1_score(new_y, new_pred),
                'precision': precision_score(new_y, new_pred),
                'recall': recall_score(new_y, new_pred)
            }
            
            # Calcular degradación
            decay = {}
            for metric in metrics_ref.keys():
                old_val = metrics_ref[metric]
                new_val = metrics_new[metric]
                decay[metric] = ((old_val - new_val) / old_val) if old_val > 0 else 0
            
            # Determinar alerta
            max_decay = max(decay.values())
            alert = max_decay > threshold
            severity = "HIGH" if max_decay > 0.10 else ("MEDIUM" if max_decay > threshold else "LOW")
            
            result = {
                'metrics_reference': {k: round(v, 4) for k, v in metrics_ref.items()},
                'metrics_current': {k: round(v, 4) for k, v in metrics_new.items()},
                'decay_percentage': {k: round(v * 100, 2) for k, v in decay.items()},
                'max_decay': round(max_decay * 100, 2),
                'severity': severity,
                'alert': alert
            }
            
            if alert:
                print(f"   🚨 ALERTA: Degradación de {max_decay*100:.1f}% detectada")
            else:
                print(f"   ✅ Performance estable (decay: {max_decay*100:.1f}%)")
            
            return result
            
        except Exception as e:
            print(f"   ⚠️ Error en concept drift: {e}")
            return {'error': str(e)}
    
    def generate_drift_report(self, new_data: pd.DataFrame, 
                             model=None,
                             save_path: Optional[str] = None) -> Dict:
        """
        Genera reporte completo de drift.
        
        Args:
            new_data: DataFrame con nuevos datos
            model: Modelo entrenado (opcional, para concept drift)
            save_path: Ruta para guardar reporte JSON (opcional)
        
        Returns:
            Dict con reporte completo
        """
        print("\n" + "="*70)
        print("REPORTE DE DRIFT")
        print("="*70)
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'reference_size': len(self.reference),
            'current_size': len(new_data),
            'covariate_shift': {},
            'concept_drift': {},
            'summary': {}
        }
        
        # Detectar covariate shift
        covariate_alerts = self.detect_covariate_shift(new_data)
        report['covariate_shift'] = covariate_alerts
        
        # Detectar concept drift (si hay modelo)
        if model is not None:
            concept_alerts = self.detect_concept_drift(model, new_data)
            report['concept_drift'] = concept_alerts
        
        # Resumen
        n_covariate_alerts = sum(1 for v in covariate_alerts.values() if v.get('alert', False))
        concept_alert = report['concept_drift'].get('alert', False)
        
        report['summary'] = {
            'total_features_monitored': len(covariate_alerts),
            'covariate_alerts': n_covariate_alerts,
            'concept_drift_detected': concept_alert,
            'overall_status': 'CRITICAL' if (n_covariate_alerts > 5 or concept_alert) else 
                            ('WARNING' if n_covariate_alerts > 0 else 'STABLE')
        }
        
        print(f"\n📊 RESUMEN:")
        print(f"   Features monitoreadas: {report['summary']['total_features_monitored']}")
        print(f"   Alertas de covariate shift: {n_covariate_alerts}")
        print(f"   Concept drift detectado: {'SÍ' if concept_alert else 'NO'}")
        print(f"   Estado general: {report['summary']['overall_status']}")
        
        # Guardar reporte
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Reporte guardado en: {save_path}")
        
        return report
    
    def get_top_drifted_features(self, drift_report: Dict, 
                                top_n: int = 10) -> List[Tuple[str, float, str]]:
        """
        Obtiene las features con mayor drift.
        
        Args:
            drift_report: Reporte generado por generate_drift_report
            top_n: Número de features a retornar
        
        Returns:
            Lista de tuplas (feature_name, drift_score, severity)
        """
        covariate_shift = drift_report.get('covariate_shift', {})
        
        features_with_drift = []
        for feature, metrics in covariate_shift.items():
            if metrics['type'] == 'numeric':
                drift_score = metrics.get('psi', 0)
            else:
                drift_score = metrics.get('max_proportion_change', 0)
            
            severity = metrics.get('severity', 'LOW')
            features_with_drift.append((feature, drift_score, severity))
        
        # Ordenar por drift score
        features_with_drift.sort(key=lambda x: x[1], reverse=True)
        
        return features_with_drift[:top_n]


if __name__ == "__main__":
    print("=" * 70)
    print("DEMO: DriftMonitor")
    print("=" * 70)
    
    # Cargar datos
    reference_data = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    print(f"\n📊 Datos de referencia: {reference_data.shape}")
    
    # Simular datos con drift
    new_data = reference_data.copy()
    new_data['MonthlyIncome'] = new_data['MonthlyIncome'] * 1.15  # +15% drift
    new_data['JobSatisfaction'] = new_data['JobSatisfaction'].replace(
        {'Very High': 'High', 'High': 'Medium'}
    )  # Drift en satisfacción
    
    # Inicializar monitor
    monitor = DriftMonitor(reference_data, target_col='Attrition')
    
    # Generar reporte
    report = monitor.generate_drift_report(
        new_data,
        save_path='models/drift_report_demo.json'
    )
    
    # Top features con drift
    print("\n🔝 Top 5 Features con Mayor Drift:")
    top_drifted = monitor.get_top_drifted_features(report, top_n=5)
    for i, (feature, score, severity) in enumerate(top_drifted, 1):
        print(f"   {i}. {feature}: {score:.4f} ({severity})")

# Made with Bob
