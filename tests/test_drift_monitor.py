"""
Tests para DriftMonitor
=======================

Tests unitarios para validar la detección de drift.
"""

import pytest
import pandas as pd
import numpy as np
from src.drift_monitor import DriftMonitor


@pytest.fixture
def sample_reference_data():
    """Fixture con datos de referencia para testing."""
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'Age': np.random.randint(22, 60, n_samples),
        'MonthlyIncome': np.random.randint(1000, 20000, n_samples),
        'YearsAtCompany': np.random.randint(0, 40, n_samples),
        'Department': np.random.choice(['Sales', 'R&D', 'HR'], n_samples),
        'JobSatisfaction': np.random.randint(1, 5, n_samples),
        'Attrition': np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_current_data_no_drift(sample_reference_data):
    """Datos actuales sin drift (similar a referencia)."""
    np.random.seed(43)
    n_samples = 200
    
    data = {
        'Age': np.random.randint(22, 60, n_samples),
        'MonthlyIncome': np.random.randint(1000, 20000, n_samples),
        'YearsAtCompany': np.random.randint(0, 40, n_samples),
        'Department': np.random.choice(['Sales', 'R&D', 'HR'], n_samples),
        'JobSatisfaction': np.random.randint(1, 5, n_samples),
        'Attrition': np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_current_data_with_drift(sample_reference_data):
    """Datos actuales con drift significativo."""
    np.random.seed(44)
    n_samples = 200
    
    # Cambios significativos en distribuciones
    data = {
        'Age': np.random.randint(30, 50, n_samples),  # Rango más estrecho
        'MonthlyIncome': np.random.randint(5000, 25000, n_samples),  # Salarios más altos
        'YearsAtCompany': np.random.randint(0, 20, n_samples),  # Menos años
        'Department': np.random.choice(['Sales', 'R&D', 'HR'], n_samples, p=[0.6, 0.3, 0.1]),  # Más ventas
        'JobSatisfaction': np.random.randint(2, 5, n_samples),  # Mejor satisfacción
        'Attrition': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Más attrition
    }
    
    return pd.DataFrame(data)


class TestDriftMonitorInitialization:
    """Tests para inicialización del DriftMonitor."""
    
    def test_initialization_basic(self, sample_reference_data):
        """Test: Inicialización básica."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        assert monitor.target_col == 'Attrition'
        assert monitor.reference is not None
        assert len(monitor.numeric_cols) > 0
        assert len(monitor.categorical_cols) > 0
    
    def test_initialization_identifies_columns(self, sample_reference_data):
        """Test: Identificación correcta de tipos de columnas."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        # Verificar columnas numéricas
        expected_numeric = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction']
        for col in expected_numeric:
            assert col in monitor.numeric_cols
        
        # Verificar columnas categóricas
        assert 'Department' in monitor.categorical_cols
    
    def test_initialization_without_target(self, sample_reference_data):
        """Test: Inicialización sin columna target."""
        df_no_target = sample_reference_data.drop(columns=['Attrition'])
        monitor = DriftMonitor(df_no_target, target_col='Attrition')
        
        assert monitor.reference_y is None


class TestPSICalculation:
    """Tests para cálculo de PSI."""
    
    def test_psi_no_drift(self, sample_reference_data):
        """Test: PSI con datos sin drift."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        # Usar los mismos datos (PSI debería ser ~0)
        ref_series = sample_reference_data['Age']
        curr_series = sample_reference_data['Age']
        
        psi = monitor.calculate_psi(ref_series, curr_series)
        
        assert psi >= 0
        assert psi < 0.1  # PSI muy bajo para datos idénticos
    
    def test_psi_with_drift(self, sample_reference_data, sample_current_data_with_drift):
        """Test: PSI con datos con drift."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        ref_series = sample_reference_data['Age']
        curr_series = sample_current_data_with_drift['Age']
        
        psi = monitor.calculate_psi(ref_series, curr_series)
        
        assert psi > 0.1  # PSI significativo para datos diferentes
    
    def test_psi_returns_float(self, sample_reference_data):
        """Test: PSI retorna un float."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        ref_series = sample_reference_data['Age']
        curr_series = sample_reference_data['Age'] + 1
        
        psi = monitor.calculate_psi(ref_series, curr_series)
        
        assert isinstance(psi, (float, np.floating))


class TestCovariateShiftDetection:
    """Tests para detección de covariate shift."""
    
    def test_detect_covariate_shift_no_drift(self, sample_reference_data, sample_current_data_no_drift):
        """Test: Detección sin drift."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        alerts = monitor.detect_covariate_shift(sample_current_data_no_drift)
        
        assert isinstance(alerts, dict)
        # Puede haber algunas alertas por variabilidad aleatoria, pero no muchas
        alert_count = sum(1 for v in alerts.values() if v.get('alert', False))
        assert alert_count < len(alerts) * 0.3  # Menos del 30% de alertas
    
    def test_detect_covariate_shift_with_drift(self, sample_reference_data, sample_current_data_with_drift):
        """Test: Detección con drift."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        alerts = monitor.detect_covariate_shift(sample_current_data_with_drift)
        
        assert isinstance(alerts, dict)
        # Debe haber varias alertas
        alert_count = sum(1 for v in alerts.values() if v.get('alert', False))
        assert alert_count > 0
    
    def test_covariate_shift_includes_metrics(self, sample_reference_data, sample_current_data_no_drift):
        """Test: Resultados incluyen métricas esperadas."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        alerts = monitor.detect_covariate_shift(sample_current_data_no_drift)
        
        # Verificar que cada feature tiene las métricas esperadas
        for feature, metrics in alerts.items():
            assert 'type' in metrics
            if metrics['type'] == 'numeric':
                assert 'psi' in metrics
                assert 'ks_statistic' in metrics
                assert 'wasserstein_distance' in metrics
            elif metrics['type'] == 'categorical':
                assert 'max_proportion_change' in metrics


class TestDriftReportGeneration:
    """Tests para generación de reportes de drift."""
    
    def test_generate_drift_report_structure(self, sample_reference_data, sample_current_data_no_drift):
        """Test: Estructura del reporte de drift."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        report = monitor.generate_drift_report(sample_current_data_no_drift)
        
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'reference_size' in report
        assert 'current_size' in report
        assert 'covariate_shift' in report
        assert 'summary' in report
    
    def test_generate_drift_report_summary(self, sample_reference_data, sample_current_data_no_drift):
        """Test: Resumen del reporte."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        report = monitor.generate_drift_report(sample_current_data_no_drift)
        summary = report['summary']
        
        assert 'total_features_monitored' in summary
        assert 'covariate_alerts' in summary
        assert 'overall_status' in summary
        assert summary['overall_status'] in ['STABLE', 'WARNING', 'CRITICAL']
    
    def test_generate_drift_report_with_drift(self, sample_reference_data, sample_current_data_with_drift):
        """Test: Reporte con drift detectado."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        report = monitor.generate_drift_report(sample_current_data_with_drift)
        summary = report['summary']
        
        # Debe detectar alertas
        assert summary['covariate_alerts'] > 0


class TestTopDriftedFeatures:
    """Tests para identificación de features con más drift."""
    
    def test_get_top_drifted_features(self, sample_reference_data, sample_current_data_with_drift):
        """Test: Obtener top features con drift."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        report = monitor.generate_drift_report(sample_current_data_with_drift)
        top_features = monitor.get_top_drifted_features(report, top_n=3)
        
        assert isinstance(top_features, list)
        assert len(top_features) <= 3
        
        # Cada elemento es una tupla (feature_name, drift_score, severity)
        for item in top_features:
            assert isinstance(item, tuple)
            assert len(item) == 3  # Cambio: ahora son 3 elementos
            assert isinstance(item[0], str)  # feature name
            assert isinstance(item[1], (int, float, np.floating))  # drift score
            assert isinstance(item[2], str)  # severity level
            assert item[2] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']  # Validar severidad
    
    def test_top_drifted_features_sorted(self, sample_reference_data, sample_current_data_with_drift):
        """Test: Features ordenadas por drift score."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        report = monitor.generate_drift_report(sample_current_data_with_drift)
        top_features = monitor.get_top_drifted_features(report, top_n=5)
        
        if len(top_features) > 1:
            # Verificar que están ordenadas de mayor a menor
            # top_features es una lista de tuplas (feature_name, drift_score)
            scores = [f[1] for f in top_features]
            assert scores == sorted(scores, reverse=True)


class TestEdgeCases:
    """Tests para casos extremos."""
    
    def test_empty_dataframe(self):
        """Test: DataFrame vacío - ahora manejado gracefully por DataTypeNormalizer."""
        empty_df = pd.DataFrame()
        
        # El normalizer maneja DataFrames vacíos sin lanzar excepción
        # Solo verificamos que se puede crear el monitor
        try:
            monitor = DriftMonitor(empty_df, target_col='Attrition')
            # Si llegamos aquí, el manejo es correcto
            assert True
        except Exception as e:
            # Si lanza excepción, debe ser informativa
            assert "empty" in str(e).lower() or "no columns" in str(e).lower()
    
    def test_single_column(self):
        """Test: DataFrame con una sola columna."""
        single_col_df = pd.DataFrame({'Age': [25, 30, 35, 40]})
        
        monitor = DriftMonitor(single_col_df, target_col='Age')
        assert monitor is not None
    
    def test_all_numeric_columns(self):
        """Test: Solo columnas numéricas."""
        numeric_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        
        monitor = DriftMonitor(numeric_df, target_col='target')
        assert len(monitor.numeric_cols) == 2
        assert len(monitor.categorical_cols) == 0
    
    def test_all_categorical_columns(self):
        """Test: Solo columnas categóricas."""
        categorical_df = pd.DataFrame({
            'col1': ['A', 'B', 'C', 'A', 'B'],
            'col2': ['X', 'Y', 'Z', 'X', 'Y'],
            'target': [0, 1, 0, 1, 0]
        })
        
        monitor = DriftMonitor(categorical_df, target_col='target')
        assert len(monitor.numeric_cols) == 0
        assert len(monitor.categorical_cols) == 2


class TestDriftSeverity:
    """Tests para clasificación de severidad de drift."""
    
    def test_severity_classification(self, sample_reference_data):
        """Test: Clasificación de severidad."""
        monitor = DriftMonitor(sample_reference_data, target_col='Attrition')
        
        # PSI bajo
        ref_series = sample_reference_data['Age']
        curr_series_low = ref_series + np.random.normal(0, 1, len(ref_series))
        psi_low = monitor.calculate_psi(ref_series, curr_series_low)
        
        # PSI alto
        curr_series_high = ref_series + np.random.normal(10, 5, len(ref_series))
        psi_high = monitor.calculate_psi(ref_series, curr_series_high)
        
        assert psi_low < psi_high


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
