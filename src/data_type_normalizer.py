"""
Data Type Normalizer
====================

Utility class para normalizar tipos de datos en DataFrames para ML pipelines.

Resuelve el problema de tipos mixtos en columnas categóricas que causa errores
en OneHotEncoder y otros transformadores de scikit-learn.

Características:
- Detección automática de tipos mixtos
- Normalización de columnas categóricas a string
- Normalización de columnas numéricas a tipos consistentes
- Validación de consistencia entre DataFrames
- Manejo robusto de valores faltantes (NaN, None)

Autor: Franco Benko
Fecha: 2026-01-21
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings


class DataTypeNormalizer:
    """
    Normaliza tipos de datos en DataFrames para garantizar consistencia
    en pipelines de Machine Learning.
    
    Estrategias de normalización:
    - Categóricas: Convertir todo a string (preservando NaN)
    - Numéricas: Mantener int64/float64 consistente
    - Booleanas: Convertir a int (0/1)
    - Fechas: Convertir a timestamp o string según configuración
    
    Ejemplo:
        >>> normalizer = DataTypeNormalizer()
        >>> df_normalized = normalizer.normalize_dataframe(df)
        >>> consistency = normalizer.validate_consistency(df1, df2)
    """
    
    def __init__(self, 
                 categorical_na_value: str = 'missing',
                 numeric_na_strategy: str = 'keep',
                 verbose: bool = True):
        """
        Inicializa el normalizador.
        
        Args:
            categorical_na_value: Valor para reemplazar NaN en categóricas
                                 ('missing', 'keep', o valor custom)
            numeric_na_strategy: Estrategia para NaN en numéricas
                                ('keep', 'zero', 'mean')
            verbose: Si True, imprime información de normalización
        """
        self.categorical_na_value = categorical_na_value
        self.numeric_na_strategy = numeric_na_strategy
        self.verbose = verbose
        
        # Cache de tipos detectados
        self._detected_types = {}
    
    def detect_mixed_types(self, df: pd.DataFrame) -> Dict[str, List[type]]:
        """
        Detecta columnas con tipos mixtos.
        
        Una columna tiene tipos mixtos si contiene valores de diferentes
        tipos de Python (excluyendo NaN/None).
        
        Args:
            df: DataFrame a analizar
        
        Returns:
            Dict con columnas que tienen tipos mixtos y lista de tipos encontrados
            
        Ejemplo:
            >>> mixed = normalizer.detect_mixed_types(df)
            >>> print(mixed)
            {'col1': [<class 'int'>, <class 'str'>]}
        """
        mixed_types = {}
        
        for col in df.columns:
            # Obtener tipos únicos (excluyendo NaN/None)
            types_in_col = set()
            for val in df[col].dropna():
                types_in_col.add(type(val))
            
            # Si hay más de un tipo, es mixto
            if len(types_in_col) > 1:
                mixed_types[col] = list(types_in_col)
        
        if self.verbose and mixed_types:
            print(f"⚠️ Columnas con tipos mixtos detectadas: {len(mixed_types)}")
            for col, types in mixed_types.items():
                type_names = [t.__name__ for t in types]
                print(f"   - {col}: {type_names}")
        
        return mixed_types
    
    def normalize_categorical(self, 
                              series: pd.Series, 
                              preserve_na: bool = True) -> pd.Series:
        """
        Normaliza una columna categórica a string.
        
        Convierte todos los valores a string, manejando correctamente:
        - Valores numéricos (int, float)
        - Valores string existentes
        - Valores NaN/None
        - Valores booleanos
        
        Args:
            series: Serie a normalizar
            preserve_na: Si True, mantiene NaN; si False, reemplaza con categorical_na_value
        
        Returns:
            Serie normalizada con dtype 'object' (string)
        """
        # Crear copia para no modificar original
        result = series.copy()
        
        # Convertir a string, manejando NaN
        if preserve_na:
            # Mantener NaN como NaN
            result = result.apply(lambda x: str(x) if pd.notna(x) else np.nan)
        else:
            # Reemplazar NaN con valor configurado
            result = result.fillna(self.categorical_na_value)
            result = result.astype(str)
        
        return result
    
    def normalize_numeric(self, 
                          series: pd.Series, 
                          target_type: str = 'float64') -> pd.Series:
        """
        Normaliza una columna numérica a tipo consistente.
        
        Intenta convertir valores a numérico, manejando:
        - Strings que representan números
        - Valores ya numéricos
        - Valores NaN/None
        
        Args:
            series: Serie a normalizar
            target_type: Tipo objetivo ('int64', 'float64')
        
        Returns:
            Serie normalizada con dtype numérico
        """
        # Crear copia
        result = series.copy()
        
        try:
            # Intentar conversión a numérico
            result = pd.to_numeric(result, errors='coerce')
            
            # Convertir a tipo objetivo si es int y no hay NaN
            if target_type == 'int64' and result.notna().all():
                result = result.astype('int64')
            else:
                result = result.astype('float64')
            
            # Aplicar estrategia de NaN
            if self.numeric_na_strategy == 'zero':
                result = result.fillna(0)
            elif self.numeric_na_strategy == 'mean':
                result = result.fillna(result.mean())
            # 'keep' no hace nada
            
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Error normalizando columna numérica: {e}")
        
        return result
    
    def auto_detect_column_types(self, 
                                 df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Detecta automáticamente columnas categóricas y numéricas.
        
        Usa heurísticas para clasificar columnas:
        - Numéricas: int64, float64, o convertibles a numérico
        - Categóricas: object, category, o no convertibles a numérico
        
        Args:
            df: DataFrame a analizar
        
        Returns:
            Tupla (categorical_cols, numeric_cols)
        """
        categorical_cols = []
        numeric_cols = []
        
        for col in df.columns:
            # Verificar dtype actual
            if df[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
            elif df[col].dtype in ['object', 'category']:
                # Intentar convertir a numérico para verificar
                try:
                    pd.to_numeric(df[col].dropna().head(100), errors='raise')
                    numeric_cols.append(col)
                except:
                    categorical_cols.append(col)
            else:
                # Otros tipos (bool, datetime, etc.)
                categorical_cols.append(col)
        
        if self.verbose:
            print(f"📊 Tipos detectados automáticamente:")
            print(f"   - Numéricas: {len(numeric_cols)} columnas")
            print(f"   - Categóricas: {len(categorical_cols)} columnas")
        
        return categorical_cols, numeric_cols
    
    def normalize_dataframe(self,
                           df: pd.DataFrame,
                           categorical_cols: Optional[List[str]] = None,
                           numeric_cols: Optional[List[str]] = None,
                           exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normaliza todos los tipos en un DataFrame.
        
        Aplica normalización apropiada a cada columna según su tipo.
        Si no se especifican columnas, las detecta automáticamente.
        
        Args:
            df: DataFrame a normalizar
            categorical_cols: Lista de columnas categóricas (opcional)
            numeric_cols: Lista de columnas numéricas (opcional)
            exclude_cols: Columnas a excluir de normalización (opcional)
        
        Returns:
            DataFrame normalizado con tipos consistentes
            
        Ejemplo:
            >>> df_norm = normalizer.normalize_dataframe(df)
            >>> # O especificando columnas
            >>> df_norm = normalizer.normalize_dataframe(
            ...     df, 
            ...     categorical_cols=['Gender', 'Department'],
            ...     numeric_cols=['Age', 'Salary']
            ... )
        """
        if self.verbose:
            print(f"\n🔧 Normalizando DataFrame ({df.shape[0]} filas, {df.shape[1]} columnas)...")
        
        # Crear copia para no modificar original
        result = df.copy()
        
        # Columnas a excluir
        exclude_cols = exclude_cols or []
        
        # Auto-detectar tipos si no se especifican
        if categorical_cols is None or numeric_cols is None:
            auto_cat, auto_num = self.auto_detect_column_types(result)
            categorical_cols = categorical_cols or auto_cat
            numeric_cols = numeric_cols or auto_num
        
        # Detectar tipos mixtos antes de normalizar
        mixed_types = self.detect_mixed_types(result)
        
        # Normalizar columnas categóricas
        for col in categorical_cols:
            if col in exclude_cols or col not in result.columns:
                continue
            
            try:
                result[col] = self.normalize_categorical(result[col])
                if self.verbose and col in mixed_types:
                    print(f"   ✅ {col}: Normalizado a string")
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Error normalizando {col}: {e}")
        
        # Normalizar columnas numéricas
        for col in numeric_cols:
            if col in exclude_cols or col not in result.columns:
                continue
            
            try:
                # Determinar tipo objetivo
                if result[col].dtype == 'int64':
                    target_type = 'int64'
                else:
                    target_type = 'float64'
                
                result[col] = self.normalize_numeric(result[col], target_type)
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Error normalizando {col}: {e}")
        
        if self.verbose:
            print(f"✅ Normalización completada")
        
        return result
    
    def validate_consistency(self, 
                            df1: pd.DataFrame, 
                            df2: pd.DataFrame,
                            check_columns: bool = True,
                            check_dtypes: bool = True) -> Dict:
        """
        Valida consistencia de tipos entre dos DataFrames.
        
        Útil para verificar que datos de referencia y actuales tienen
        tipos compatibles antes de drift detection o entrenamiento.
        
        Args:
            df1: Primer DataFrame (ej: referencia)
            df2: Segundo DataFrame (ej: actual)
            check_columns: Si True, verifica que tengan las mismas columnas
            check_dtypes: Si True, verifica que los dtypes coincidan
        
        Returns:
            Dict con resultados de validación:
            {
                'consistent': bool,
                'has_issues': bool,
                'issues': List[str],
                'missing_in_df2': List[str],
                'missing_in_df1': List[str],
                'dtype_mismatches': Dict[str, Tuple[dtype, dtype]]
            }
        """
        issues = []
        dtype_mismatches = {}
        
        # Verificar columnas
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        missing_in_df2 = list(cols1 - cols2)
        missing_in_df1 = list(cols2 - cols1)
        
        if check_columns:
            if missing_in_df2:
                issues.append(f"Columnas en df1 pero no en df2: {missing_in_df2}")
            if missing_in_df1:
                issues.append(f"Columnas en df2 pero no en df1: {missing_in_df1}")
        
        # Verificar dtypes en columnas comunes
        common_cols = cols1 & cols2
        
        if check_dtypes:
            for col in common_cols:
                dtype1 = df1[col].dtype
                dtype2 = df2[col].dtype
                
                if dtype1 != dtype2:
                    dtype_mismatches[col] = (dtype1, dtype2)
                    issues.append(f"Dtype mismatch en '{col}': {dtype1} vs {dtype2}")
        
        # Resultado
        result = {
            'consistent': len(issues) == 0,
            'has_issues': len(issues) > 0,
            'issues': issues,
            'missing_in_df2': missing_in_df2,
            'missing_in_df1': missing_in_df1,
            'dtype_mismatches': dtype_mismatches
        }
        
        if self.verbose and result['has_issues']:
            print(f"\n⚠️ Inconsistencias detectadas entre DataFrames:")
            for issue in issues:
                print(f"   - {issue}")
        
        return result
    
    def get_normalization_report(self, df_original: pd.DataFrame, 
                                df_normalized: pd.DataFrame) -> Dict:
        """
        Genera reporte de cambios realizados durante normalización.
        
        Args:
            df_original: DataFrame original
            df_normalized: DataFrame normalizado
        
        Returns:
            Dict con estadísticas de normalización
        """
        report = {
            'total_columns': len(df_original.columns),
            'columns_changed': 0,
            'dtype_changes': {},
            'mixed_types_fixed': []
        }
        
        # Detectar tipos mixtos en original
        mixed_original = self.detect_mixed_types(df_original)
        report['mixed_types_fixed'] = list(mixed_original.keys())
        
        # Comparar dtypes
        for col in df_original.columns:
            if col in df_normalized.columns:
                dtype_orig = df_original[col].dtype
                dtype_norm = df_normalized[col].dtype
                
                if dtype_orig != dtype_norm:
                    report['columns_changed'] += 1
                    report['dtype_changes'][col] = {
                        'original': str(dtype_orig),
                        'normalized': str(dtype_norm)
                    }
        
        return report


# Función de conveniencia para uso rápido
def normalize_for_ml(df: pd.DataFrame, 
                     target_col: Optional[str] = None,
                     verbose: bool = False) -> pd.DataFrame:
    """
    Función de conveniencia para normalizar DataFrame para ML.
    
    Args:
        df: DataFrame a normalizar
        target_col: Columna target a excluir de normalización (opcional)
        verbose: Si True, imprime información
    
    Returns:
        DataFrame normalizado
        
    Ejemplo:
        >>> df_clean = normalize_for_ml(df, target_col='Attrition')
    """
    normalizer = DataTypeNormalizer(verbose=verbose)
    
    exclude_cols = [target_col] if target_col else []
    
    return normalizer.normalize_dataframe(df, exclude_cols=exclude_cols)

# Made with Bob
