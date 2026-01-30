"""
Validación Temporal con Walk-Forward
====================================

Implementa estrategias de validación temporal para evitar data leakage
y evaluar correctamente modelos en series temporales.

Características:
- Walk-Forward Validation
- Expanding Window
- Rolling Window
- Métricas por período temporal

Autor: IBM Bob (Data Science Expert)
Fecha: 2026-01-20
"""

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class TemporalValidator:
    """
    Validador temporal para modelos de ML.
    
    Implementa walk-forward validation respetando el orden temporal
    de los datos para evitar data leakage.
    """
    
    def __init__(self, date_column: str = 'DataMonth', target_column: str = 'Attrition'):
        """
        Inicializa el validador temporal.
        
        Args:
            date_column: Nombre de la columna con fechas
            target_column: Nombre de la columna target
        """
        self.date_column = date_column
        self.target_column = target_column
    
    def walk_forward_validation(self, 
                               data: pd.DataFrame,
                               model,
                               n_splits: int = 5,
                               strategy: str = 'expanding') -> Dict:
        """
        Realiza walk-forward validation.
        
        Estrategias:
        - 'expanding': Ventana de entrenamiento crece (acumula datos)
        - 'rolling': Ventana de entrenamiento fija (sliding window)
        
        Args:
            data: DataFrame con datos temporales
            model: Modelo a validar (debe tener fit/predict)
            n_splits: Número de splits temporales
            strategy: 'expanding' o 'rolling'
        
        Returns:
            Dict con resultados de validación
        """
        print(f"\n🔄 Walk-Forward Validation ({strategy} window)")
        print(f"   Splits: {n_splits}")
        
        # Ordenar por fecha
        data = data.sort_values(self.date_column).reset_index(drop=True)
        
        # Obtener períodos únicos
        unique_periods = sorted(data[self.date_column].unique())
        
        if len(unique_periods) < n_splits + 1:
            raise ValueError(f"No hay suficientes períodos. Necesitas al menos {n_splits + 1}")
        
        # Preparar features y target
        X = data.drop(columns=[self.target_column, self.date_column])
        y = data[self.target_column]
        
        # Convertir columnas categóricas a string para evitar errores de encoding
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].astype(str)
        
        # Convertir target a binario si es necesario
        if y.dtype == 'object':
            y = y.map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
        
        results = []
        
        # Calcular tamaño de ventana para rolling
        if strategy == 'rolling':
            window_size = len(unique_periods) // (n_splits + 1)
        
        for i in range(n_splits):
            # Definir períodos de train y test
            if strategy == 'expanding':
                # Ventana creciente: train desde inicio hasta período i
                train_end_idx = i + 1
                train_periods = unique_periods[:train_end_idx]
                test_period = unique_periods[train_end_idx]
            else:  # rolling
                # Ventana fija: últimos window_size períodos
                train_start_idx = max(0, i + 1 - window_size)
                train_end_idx = i + 1
                train_periods = unique_periods[train_start_idx:train_end_idx]
                test_period = unique_periods[train_end_idx]
            
            # Crear máscaras
            train_mask = data[self.date_column].isin(train_periods)
            test_mask = data[self.date_column] == test_period
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
            
            # Entrenar modelo
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            
            # Predecir
            y_pred = model_clone.predict(X_test)
            y_proba = model_clone.predict_proba(X_test)[:, 1]
            
            # Calcular métricas
            metrics = {
                'split': i + 1,
                'train_periods': [str(p) for p in train_periods],
                'test_period': str(test_period),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'f1_score': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred)
            }
            
            results.append(metrics)
            
            print(f"\n   Split {i+1}/{n_splits}:")
            print(f"      Train: {len(train_periods)} períodos ({len(X_train)} registros)")
            print(f"      Test: {test_period} ({len(X_test)} registros)")
            print(f"      ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"      F1-Score: {metrics['f1_score']:.4f}")
        
        # Calcular estadísticas agregadas
        summary = {
            'strategy': strategy,
            'n_splits': n_splits,
            'results_by_split': results,
            'mean_metrics': {
                'roc_auc': np.mean([r['roc_auc'] for r in results]),
                'f1_score': np.mean([r['f1_score'] for r in results]),
                'precision': np.mean([r['precision'] for r in results]),
                'recall': np.mean([r['recall'] for r in results])
            },
            'std_metrics': {
                'roc_auc': np.std([r['roc_auc'] for r in results]),
                'f1_score': np.std([r['f1_score'] for r in results]),
                'precision': np.std([r['precision'] for r in results]),
                'recall': np.std([r['recall'] for r in results])
            }
        }
        
        print(f"\n📊 RESUMEN:")
        print(f"   ROC-AUC: {summary['mean_metrics']['roc_auc']:.4f} ± {summary['std_metrics']['roc_auc']:.4f}")
        print(f"   F1-Score: {summary['mean_metrics']['f1_score']:.4f} ± {summary['std_metrics']['f1_score']:.4f}")
        
        return summary
    
    def compare_validation_strategies(self,
                                     data: pd.DataFrame,
                                     model,
                                     n_splits: int = 5) -> Dict:
        """
        Compara diferentes estrategias de validación.
        
        Args:
            data: DataFrame con datos temporales
            model: Modelo a validar
            n_splits: Número de splits
        
        Returns:
            Dict con comparación de estrategias
        """
        print("\n" + "="*70)
        print("COMPARACIÓN DE ESTRATEGIAS DE VALIDACIÓN")
        print("="*70)
        
        results = {}
        
        # Walk-forward expanding
        results['walk_forward_expanding'] = self.walk_forward_validation(
            data, model, n_splits, strategy='expanding'
        )
        
        # Walk-forward rolling
        results['walk_forward_rolling'] = self.walk_forward_validation(
            data, model, n_splits, strategy='rolling'
        )
        
        # Random split (para comparación - NO RECOMENDADO para temporal)
        print(f"\n🔄 Random Split (baseline - NO temporal)")
        from sklearn.model_selection import cross_val_score
        
        X = data.drop(columns=[self.target_column, self.date_column])
        y = data[self.target_column]
        if y.dtype == 'object':
            y = y.map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
        
        cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring='roc_auc')
        
        results['random_split'] = {
            'strategy': 'random',
            'mean_metrics': {'roc_auc': np.mean(cv_scores)},
            'std_metrics': {'roc_auc': np.std(cv_scores)}
        }
        
        print(f"   ROC-AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Comparación
        print(f"\n📊 COMPARACIÓN:")
        print(f"   Random Split:      {results['random_split']['mean_metrics']['roc_auc']:.4f}")
        print(f"   Walk-Forward Exp:  {results['walk_forward_expanding']['mean_metrics']['roc_auc']:.4f}")
        print(f"   Walk-Forward Roll: {results['walk_forward_rolling']['mean_metrics']['roc_auc']:.4f}")
        
        # Advertencia si random es muy diferente
        random_auc = results['random_split']['mean_metrics']['roc_auc']
        temporal_auc = results['walk_forward_expanding']['mean_metrics']['roc_auc']
        diff = abs(random_auc - temporal_auc)
        
        if diff > 0.05:
            print(f"\n⚠️ ADVERTENCIA: Diferencia significativa ({diff:.4f}) entre random y temporal")
            print(f"   Esto indica posible DATA LEAKAGE en validación random")
        
        return results
    
    def detect_performance_decay(self, 
                                data: pd.DataFrame,
                                model,
                                window_size: int = 3) -> pd.DataFrame:
        """
        Detecta degradación de performance a lo largo del tiempo.
        
        Args:
            data: DataFrame con datos temporales
            model: Modelo entrenado
            window_size: Tamaño de ventana para calcular métricas
        
        Returns:
            DataFrame con métricas por período
        """
        print(f"\n📉 Detectando Performance Decay (ventana: {window_size} períodos)")
        
        # Ordenar por fecha
        data = data.sort_values(self.date_column).reset_index(drop=True)
        unique_periods = sorted(data[self.date_column].unique())
        
        # Preparar features y target
        X = data.drop(columns=[self.target_column, self.date_column])
        y = data[self.target_column]
        if y.dtype == 'object':
            y = y.map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
        
        decay_results = []
        
        for i, period in enumerate(unique_periods):
            # Datos del período
            period_mask = data[self.date_column] == period
            X_period = X[period_mask]
            y_period = y[period_mask]
            
            if len(X_period) == 0:
                continue
            
            # Predecir
            try:
                y_pred = model.predict(X_period)
                y_proba = model.predict_proba(X_period)[:, 1]
                
                # Calcular métricas
                metrics = {
                    'period': str(period),
                    'period_index': i,
                    'n_samples': len(X_period),
                    'roc_auc': roc_auc_score(y_period, y_proba),
                    'f1_score': f1_score(y_period, y_pred),
                    'precision': precision_score(y_period, y_pred),
                    'recall': recall_score(y_period, y_pred)
                }
                
                decay_results.append(metrics)
            except Exception as e:
                print(f"   ⚠️ Error en período {period}: {e}")
        
        df_decay = pd.DataFrame(decay_results)
        
        # Calcular decay respecto al primer período
        if len(df_decay) > 0:
            baseline_auc = df_decay.iloc[0]['roc_auc']
            df_decay['auc_decay_pct'] = ((baseline_auc - df_decay['roc_auc']) / baseline_auc * 100)
            
            # Calcular tendencia (regresión lineal simple)
            x = df_decay['period_index'].values
            y_auc = df_decay['roc_auc'].values
            slope = np.polyfit(x, y_auc, 1)[0]
            
            print(f"\n   Baseline ROC-AUC: {baseline_auc:.4f}")
            print(f"   Último ROC-AUC: {df_decay.iloc[-1]['roc_auc']:.4f}")
            print(f"   Decay total: {df_decay.iloc[-1]['auc_decay_pct']:.2f}%")
            print(f"   Tendencia (slope): {slope:.6f}")
            
            if slope < -0.01:
                print(f"   🚨 ALERTA: Tendencia negativa detectada")
            elif slope > 0.01:
                print(f"   ✅ Tendencia positiva (modelo mejora)")
            else:
                print(f"   ✅ Performance estable")
        
        return df_decay


if __name__ == "__main__":
    print("=" * 70)
    print("DEMO: TemporalValidator")
    print("=" * 70)
    
    # Generar datos temporales de ejemplo
    from src.temporal_generator import TemporalHRGenerator
    
    # Cargar datos semilla
    seed_data = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    
    # Generar secuencia temporal
    generator = TemporalHRGenerator(seed_data, start_date="2024-01-01")
    temporal_data = generator.generate_temporal_sequence(n_months=6, retention_rate=0.85)
    
    print(f"\n📊 Datos temporales generados: {temporal_data.shape}")
    print(f"   Períodos: {temporal_data['DataMonth'].nunique()}")
    
    # Crear modelo simple
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    # Preparar pipeline
    numeric_cols = temporal_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['Attrition', 'DataMonth', 'EmployeeNumber']]
    
    categorical_cols = temporal_data.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ['Attrition', 'DataMonth']]
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    # Inicializar validador
    validator = TemporalValidator(date_column='DataMonth', target_column='Attrition')
    
    # Comparar estrategias
    comparison = validator.compare_validation_strategies(
        temporal_data, model, n_splits=4
    )
    
    # Detectar decay (entrenar en primeros 3 meses, evaluar en todos)
    first_3_months = temporal_data[temporal_data['DataMonth'] <= temporal_data['DataMonth'].unique()[2]]
    X_train = first_3_months.drop(columns=['Attrition', 'DataMonth'])
    y_train = first_3_months['Attrition'].map({'Yes': 1, 'No': 0})
    
    model.fit(X_train, y_train)
    
    decay_df = validator.detect_performance_decay(temporal_data, model)
    
    print("\n📊 Performance por período:")
    print(decay_df[['period', 'roc_auc', 'f1_score', 'auc_decay_pct']].to_string(index=False))

# Made with Bob
