"""
Tests para DataTypeNormalizer
==============================

Tests unitarios completos para la clase DataTypeNormalizer que maneja
la normalización de tipos de datos en DataFrames.

Autor: Franco Benko
Fecha: 2026-01-21
"""

import pytest
import pandas as pd
import numpy as np
from src.data_type_normalizer import DataTypeNormalizer, normalize_for_ml


class TestDataTypeNormalizer:
    """Tests para la clase DataTypeNormalizer."""
    
    @pytest.fixture
    def normalizer(self):
        """Fixture que retorna una instancia del normalizer."""
        return DataTypeNormalizer(verbose=False)
    
    @pytest.fixture
    def sample_df_mixed_types(self):
        """DataFrame con tipos mixtos para testing."""
        return pd.DataFrame({
            'col_mixed': [1, 2, 'three', 4, 'five'],
            'col_numeric': [1.0, 2.0, 3.0, 4.0, 5.0],
            'col_string': ['a', 'b', 'c', 'd', 'e'],
            'col_with_nan': [1, np.nan, 3, None, 5]
        })
    
    @pytest.fixture
    def sample_df_clean(self):
        """DataFrame con tipos consistentes."""
        return pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['IT', 'HR', 'Sales', 'IT', 'HR'],
            'active': [True, True, False, True, False]
        })
    
    # ==================== Tests de Detección ====================
    
    def test_detect_mixed_types_finds_mixed_column(self, normalizer, sample_df_mixed_types):
        """Test que detecta columnas con tipos mixtos."""
        mixed = normalizer.detect_mixed_types(sample_df_mixed_types)
        
        assert 'col_mixed' in mixed
        assert len(mixed['col_mixed']) == 2  # int y str
        assert int in mixed['col_mixed']
        assert str in mixed['col_mixed']
    
    def test_detect_mixed_types_ignores_clean_columns(self, normalizer, sample_df_mixed_types):
        """Test que no detecta tipos mixtos en columnas limpias."""
        mixed = normalizer.detect_mixed_types(sample_df_mixed_types)
        
        assert 'col_numeric' not in mixed
        assert 'col_string' not in mixed
    
    def test_detect_mixed_types_empty_dataframe(self, normalizer):
        """Test con DataFrame vacío."""
        df_empty = pd.DataFrame()
        mixed = normalizer.detect_mixed_types(df_empty)
        
        assert mixed == {}
    
    def test_detect_mixed_types_ignores_nan(self, normalizer, sample_df_mixed_types):
        """Test que ignora NaN al detectar tipos mixtos."""
        mixed = normalizer.detect_mixed_types(sample_df_mixed_types)
        
        # col_with_nan tiene int y NaN, pero NaN se ignora
        # Así que solo debería tener un tipo (int o float)
        if 'col_with_nan' in mixed:
            # Si se detecta como mixto, verificar que no incluye NoneType
            assert type(None) not in mixed['col_with_nan']
    
    # ==================== Tests de Normalización Categórica ====================
    
    def test_normalize_categorical_mixed_types(self, normalizer):
        """Test normalización de columna con tipos mixtos."""
        series = pd.Series([1, 2, 'three', 4, 'five'])
        result = normalizer.normalize_categorical(series)
        
        assert result.dtype == 'object'
        assert all(isinstance(x, str) for x in result)
        assert result[0] == '1'
        assert result[2] == 'three'
    
    def test_normalize_categorical_with_nan_preserve(self, normalizer):
        """Test que preserva NaN cuando preserve_na=True."""
        series = pd.Series([1, np.nan, 'three', None])
        result = normalizer.normalize_categorical(series, preserve_na=True)
        
        assert pd.isna(result[1])
        assert pd.isna(result[3])
        assert result[0] == '1'
        assert result[2] == 'three'
    
    def test_normalize_categorical_with_nan_replace(self, normalizer):
        """Test que reemplaza NaN cuando preserve_na=False."""
        series = pd.Series([1, np.nan, 'three', None])
        result = normalizer.normalize_categorical(series, preserve_na=False)
        
        assert result[1] == 'missing'
        assert result[3] == 'missing'
        assert all(isinstance(x, str) for x in result)
    
    def test_normalize_categorical_boolean(self, normalizer):
        """Test normalización de valores booleanos."""
        series = pd.Series([True, False, True, False])
        result = normalizer.normalize_categorical(series)
        
        assert result.dtype == 'object'
        assert result[0] == 'True'
        assert result[1] == 'False'
    
    # ==================== Tests de Normalización Numérica ====================
    
    def test_normalize_numeric_clean_data(self, normalizer):
        """Test normalización de datos numéricos limpios."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = normalizer.normalize_numeric(series, target_type='int64')
        
        assert result.dtype == 'int64'
        assert list(result) == [1, 2, 3, 4, 5]
    
    def test_normalize_numeric_with_nan_keep(self, normalizer):
        """Test que mantiene NaN con estrategia 'keep'."""
        normalizer_keep = DataTypeNormalizer(numeric_na_strategy='keep', verbose=False)
        series = pd.Series([1.0, np.nan, 3.0, 4.0])
        result = normalizer_keep.normalize_numeric(series)
        
        assert pd.isna(result[1])
        assert result[0] == 1.0
    
    def test_normalize_numeric_with_nan_zero(self, normalizer):
        """Test que reemplaza NaN con 0."""
        normalizer_zero = DataTypeNormalizer(numeric_na_strategy='zero', verbose=False)
        series = pd.Series([1.0, np.nan, 3.0, 4.0])
        result = normalizer_zero.normalize_numeric(series)
        
        assert result[1] == 0.0
        assert not pd.isna(result[1])
    
    def test_normalize_numeric_with_nan_mean(self, normalizer):
        """Test que reemplaza NaN con media."""
        normalizer_mean = DataTypeNormalizer(numeric_na_strategy='mean', verbose=False)
        series = pd.Series([1.0, np.nan, 3.0, 4.0])
        result = normalizer_mean.normalize_numeric(series)
        
        # Media de [1, 3, 4] = 2.666...
        assert not pd.isna(result[1])
        assert abs(result[1] - 2.666) < 0.01
    
    def test_normalize_numeric_string_numbers(self, normalizer):
        """Test conversión de strings que representan números."""
        series = pd.Series(['1', '2', '3', '4'])
        result = normalizer.normalize_numeric(series)
        
        assert result.dtype in ['int64', 'float64']
        assert result[0] == 1.0
    
    # ==================== Tests de Auto-Detección ====================
    
    def test_auto_detect_column_types(self, normalizer, sample_df_clean):
        """Test auto-detección de tipos de columnas."""
        cat_cols, num_cols = normalizer.auto_detect_column_types(sample_df_clean)
        
        assert 'age' in num_cols
        assert 'salary' in num_cols
        assert 'department' in cat_cols
        # 'active' (bool) puede estar en cualquiera dependiendo de la implementación
    
    def test_auto_detect_empty_dataframe(self, normalizer):
        """Test auto-detección con DataFrame vacío."""
        df_empty = pd.DataFrame()
        cat_cols, num_cols = normalizer.auto_detect_column_types(df_empty)
        
        assert cat_cols == []
        assert num_cols == []
    
    # ==================== Tests de Normalización Completa ====================
    
    def test_normalize_dataframe_mixed_types(self, normalizer, sample_df_mixed_types):
        """Test normalización completa de DataFrame con tipos mixtos."""
        result = normalizer.normalize_dataframe(sample_df_mixed_types)
        
        # Verificar que col_mixed ahora es string
        assert result['col_mixed'].dtype == 'object'
        assert isinstance(result['col_mixed'][0], str)
        
        # Verificar que col_numeric sigue siendo numérica
        assert result['col_numeric'].dtype in ['int64', 'float64']
    
    def test_normalize_dataframe_with_exclude(self, normalizer, sample_df_clean):
        """Test normalización excluyendo columnas específicas."""
        result = normalizer.normalize_dataframe(
            sample_df_clean,
            exclude_cols=['department']
        )
        
        # department no debería cambiar
        assert result['department'].equals(sample_df_clean['department'])
    
    def test_normalize_dataframe_explicit_columns(self, normalizer, sample_df_clean):
        """Test normalización con columnas explícitas."""
        result = normalizer.normalize_dataframe(
            sample_df_clean,
            categorical_cols=['department'],
            numeric_cols=['age', 'salary']
        )
        
        assert result['department'].dtype == 'object'
        assert result['age'].dtype in ['int64', 'float64']
        assert result['salary'].dtype in ['int64', 'float64']
    
    def test_normalize_dataframe_preserves_shape(self, normalizer, sample_df_clean):
        """Test que la normalización preserva la forma del DataFrame."""
        result = normalizer.normalize_dataframe(sample_df_clean)
        
        assert result.shape == sample_df_clean.shape
        assert list(result.columns) == list(sample_df_clean.columns)
    
    # ==================== Tests de Validación de Consistencia ====================
    
    def test_validate_consistency_same_types(self, normalizer):
        """Test validación con tipos consistentes."""
        df1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        df2 = pd.DataFrame({'col1': [4, 5, 6], 'col2': ['d', 'e', 'f']})
        
        result = normalizer.validate_consistency(df1, df2)
        
        assert result['consistent'] == True
        assert result['has_issues'] == False
        assert len(result['issues']) == 0
    
    def test_validate_consistency_different_types(self, normalizer):
        """Test validación con tipos inconsistentes."""
        df1 = pd.DataFrame({'col1': [1, 2, 3]})
        df2 = pd.DataFrame({'col1': ['a', 'b', 'c']})
        
        result = normalizer.validate_consistency(df1, df2, check_dtypes=True)
        
        assert result['consistent'] == False
        assert result['has_issues'] == True
        assert 'col1' in result['dtype_mismatches']
    
    def test_validate_consistency_missing_columns(self, normalizer):
        """Test validación con columnas faltantes."""
        df1 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        df2 = pd.DataFrame({'col1': [5, 6]})
        
        result = normalizer.validate_consistency(df1, df2, check_columns=True)
        
        assert result['has_issues'] == True
        assert 'col2' in result['missing_in_df2']
    
    def test_validate_consistency_extra_columns(self, normalizer):
        """Test validación con columnas extra."""
        df1 = pd.DataFrame({'col1': [1, 2]})
        df2 = pd.DataFrame({'col1': [3, 4], 'col2': [5, 6]})
        
        result = normalizer.validate_consistency(df1, df2, check_columns=True)
        
        assert result['has_issues'] == True
        assert 'col2' in result['missing_in_df1']
    
    # ==================== Tests de Reporte de Normalización ====================
    
    def test_get_normalization_report(self, normalizer, sample_df_mixed_types):
        """Test generación de reporte de normalización."""
        df_normalized = normalizer.normalize_dataframe(sample_df_mixed_types)
        report = normalizer.get_normalization_report(sample_df_mixed_types, df_normalized)
        
        assert 'total_columns' in report
        assert 'columns_changed' in report
        assert 'dtype_changes' in report
        assert 'mixed_types_fixed' in report
        
        assert report['total_columns'] == len(sample_df_mixed_types.columns)
        assert 'col_mixed' in report['mixed_types_fixed']
    
    # ==================== Tests de Función de Conveniencia ====================
    
    def test_normalize_for_ml_basic(self, sample_df_clean):
        """Test función de conveniencia normalize_for_ml."""
        result = normalize_for_ml(sample_df_clean, verbose=False)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_df_clean.shape
    
    def test_normalize_for_ml_with_target(self, sample_df_clean):
        """Test normalize_for_ml excluyendo target."""
        df_with_target = sample_df_clean.copy()
        df_with_target['target'] = [0, 1, 0, 1, 0]
        
        result = normalize_for_ml(df_with_target, target_col='target', verbose=False)
        
        # Target debería mantenerse sin cambios
        assert result['target'].equals(df_with_target['target'])
    
    # ==================== Tests de Edge Cases ====================
    
    def test_single_column_dataframe(self, normalizer):
        """Test con DataFrame de una sola columna."""
        df = pd.DataFrame({'col1': [1, 2, 'three']})
        result = normalizer.normalize_dataframe(df)
        
        assert result.shape == (3, 1)
        assert result['col1'].dtype == 'object'
    
    def test_all_nan_column(self, normalizer):
        """Test con columna completamente NaN."""
        df = pd.DataFrame({'col1': [np.nan, np.nan, np.nan]})
        result = normalizer.normalize_dataframe(df)
        
        assert result.shape == df.shape
        assert all(pd.isna(result['col1']))
    
    def test_empty_strings(self, normalizer):
        """Test con strings vacíos."""
        series = pd.Series(['', 'a', '', 'b'])
        result = normalizer.normalize_categorical(series)
        
        assert result[0] == ''
        assert result[1] == 'a'
    
    def test_very_large_numbers(self, normalizer):
        """Test con números muy grandes."""
        series = pd.Series([1e10, 2e10, 3e10])
        result = normalizer.normalize_numeric(series)
        
        assert result.dtype == 'float64'
        assert result[0] == 1e10
    
    def test_negative_numbers(self, normalizer):
        """Test con números negativos."""
        series = pd.Series([-1, -2, -3, -4])
        result = normalizer.normalize_numeric(series, target_type='int64')
        
        assert result.dtype == 'int64'
        assert result[0] == -1


# ==================== Tests de Integración ====================

class TestDataTypeNormalizerIntegration:
    """Tests de integración con casos de uso reales."""
    
    def test_hr_dataset_normalization(self):
        """Test con dataset tipo HR (caso de uso real)."""
        df = pd.DataFrame({
            'Age': [25, 30, 35],
            'Department': ['IT', 'HR', 'Sales'],
            'Salary': [50000, 60000, 70000],
            'Attrition': ['Yes', 'No', 'Yes'],
            'YearsAtCompany': [2, 5, 3]
        })
        
        normalizer = DataTypeNormalizer(verbose=False)
        result = normalizer.normalize_dataframe(df, exclude_cols=['Attrition'])
        
        # Verificar tipos esperados
        assert result['Age'].dtype in ['int64', 'float64']
        assert result['Department'].dtype == 'object'
        assert result['Salary'].dtype in ['int64', 'float64']
        assert result['Attrition'].equals(df['Attrition'])  # Excluida
    
    def test_drift_detection_workflow(self):
        """Test workflow completo de drift detection."""
        # Simular datos de referencia y actuales
        ref_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0]
        })
        
        current_data = pd.DataFrame({
            'feature1': [6, 7, 8, 9, 10],
            'feature2': ['f', 'g', 'h', 'i', 'j'],
            'target': [1, 0, 1, 0, 1]
        })
        
        normalizer = DataTypeNormalizer(verbose=False)
        
        # Normalizar ambos
        ref_norm = normalizer.normalize_dataframe(ref_data, exclude_cols=['target'])
        curr_norm = normalizer.normalize_dataframe(current_data, exclude_cols=['target'])
        
        # Validar consistencia
        consistency = normalizer.validate_consistency(ref_norm, curr_norm)
        
        assert consistency['consistent'] == True
        assert ref_norm['feature1'].dtype == curr_norm['feature1'].dtype
        assert ref_norm['feature2'].dtype == curr_norm['feature2'].dtype


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

# Made with Bob
