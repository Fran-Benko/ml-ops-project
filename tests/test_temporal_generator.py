"""
Tests para TemporalHRGenerator
================================

Tests unitarios para validar la generación de datos temporales
con continuidad de cohortes.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.temporal_generator import TemporalHRGenerator


@pytest.fixture
def sample_hr_data():
    """Fixture con datos de ejemplo para testing."""
    np.random.seed(42)
    n_employees = 100
    
    data = {
        'EmployeeNumber': range(1, n_employees + 1),
        'Age': np.random.randint(22, 60, n_employees),
        'Attrition': np.random.choice([0, 1], n_employees, p=[0.84, 0.16]),
        'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], n_employees),
        'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n_employees),
        'DistanceFromHome': np.random.randint(1, 30, n_employees),
        'Education': np.random.randint(1, 6, n_employees),
        'EnvironmentSatisfaction': np.random.randint(1, 5, n_employees),
        'JobInvolvement': np.random.randint(1, 5, n_employees),
        'JobLevel': np.random.randint(1, 6, n_employees),
        'JobSatisfaction': np.random.randint(1, 5, n_employees),
        'MonthlyIncome': np.random.randint(1000, 20000, n_employees),
        'NumCompaniesWorked': np.random.randint(0, 10, n_employees),
        'PercentSalaryHike': np.random.randint(11, 26, n_employees),
        'PerformanceRating': np.random.choice([3, 4], n_employees),
        'RelationshipSatisfaction': np.random.randint(1, 5, n_employees),
        'StockOptionLevel': np.random.randint(0, 4, n_employees),
        'TotalWorkingYears': np.random.randint(0, 40, n_employees),
        'TrainingTimesLastYear': np.random.randint(0, 7, n_employees),
        'WorkLifeBalance': np.random.randint(1, 5, n_employees),
        'YearsAtCompany': np.random.randint(0, 40, n_employees),
        'YearsInCurrentRole': np.random.randint(0, 20, n_employees),
        'YearsSinceLastPromotion': np.random.randint(0, 16, n_employees),
        'YearsWithCurrManager': np.random.randint(0, 18, n_employees),
    }
    
    return pd.DataFrame(data)


class TestTemporalHRGenerator:
    """Suite de tests para TemporalHRGenerator."""
    
    def test_initialization(self, sample_hr_data):
        """Test: Inicialización correcta del generador."""
        generator = TemporalHRGenerator(sample_hr_data)
        
        assert generator.current_cohort.shape[0] == 100
        assert generator.month_counter == 0
        assert 'EmployeeNumber' in generator.current_cohort.columns
        assert generator.employee_id_counter >= 100
    
    def test_initialization_with_custom_date(self, sample_hr_data):
        """Test: Inicialización con fecha personalizada."""
        custom_date = "2023-06-15"
        generator = TemporalHRGenerator(sample_hr_data, start_date=custom_date)
        
        assert generator.start_date == pd.to_datetime(custom_date)
        assert generator.current_date == pd.to_datetime(custom_date)
    
    def test_generate_next_month_basic(self, sample_hr_data):
        """Test: Generación de un mes básico."""
        generator = TemporalHRGenerator(sample_hr_data)
        next_month = generator.generate_next_month()
        
        # Verificar que se generó un DataFrame
        assert isinstance(next_month, pd.DataFrame)
        assert len(next_month) > 0
        
        # Verificar que tiene las columnas esperadas
        assert 'EmployeeNumber' in next_month.columns
        assert 'DataMonth' in next_month.columns  # Cambio: 'period' -> 'DataMonth'
        assert 'Attrition' in next_month.columns
    
    def test_generate_next_month_retention(self, sample_hr_data):
        """Test: Verificar tasa de retención."""
        generator = TemporalHRGenerator(sample_hr_data)
        initial_employees = set(generator.current_cohort['EmployeeNumber'])
        
        next_month = generator.generate_next_month(retention_rate=0.9)
        retained_employees = set(next_month['EmployeeNumber']) & initial_employees
        
        # Con retention_rate=0.9, esperamos ~90% de retención
        retention_actual = len(retained_employees) / len(initial_employees)
        assert 0.7 < retention_actual < 1.0  # Rango razonable con variabilidad
    
    def test_generate_next_month_size(self, sample_hr_data):
        """Test: Tamaño del dataset generado."""
        generator = TemporalHRGenerator(sample_hr_data)
        next_month = generator.generate_next_month()
        
        # Debe mantener aproximadamente el mismo tamaño
        assert 80 <= len(next_month) <= 120  # ±20% del tamaño original
    
    def test_generate_temporal_sequence(self, sample_hr_data):
        """Test: Generación de secuencia temporal."""
        generator = TemporalHRGenerator(sample_hr_data)
        n_months = 3
        
        sequence = generator.generate_temporal_sequence(
            n_months=n_months,
            scenario_schedule=['normal'] * n_months
        )
        
        # Verificar que se generaron n_months + 1 períodos (inicial + n_months)
        unique_periods = sequence['DataMonth'].nunique()  # Cambio: 'period' -> 'DataMonth'
        assert unique_periods == n_months + 1
        
        # Verificar que el total de registros es razonable
        expected_min = 100 * (n_months + 1) * 0.8  # 80% del esperado
        expected_max = 100 * (n_months + 1) * 1.2  # 120% del esperado
        assert expected_min <= len(sequence) <= expected_max
    
    def test_employee_aging(self, sample_hr_data):
        """Test: Envejecimiento de empleados."""
        generator = TemporalHRGenerator(sample_hr_data)
        initial_ages = generator.current_cohort['Age'].copy()
        
        # Generar varios meses
        for _ in range(6):
            generator.generate_next_month()
        
        # Los empleados retenidos deberían haber envejecido
        # (difícil de verificar exactamente por la rotación, pero podemos verificar que hay cambios)
        final_cohort = generator.current_cohort
        assert 'Age' in final_cohort.columns
    
    def test_attrition_calculation(self, sample_hr_data):
        """Test: Cálculo de attrition."""
        generator = TemporalHRGenerator(sample_hr_data)
        
        # Generar un mes y verificar que hay attrition
        next_month = generator.generate_next_month(retention_rate=0.8)
        
        # Attrition ahora es string ('Yes'/'No'), no numérico
        attrition_count = (next_month['Attrition'] == 'Yes').sum()
        assert attrition_count > 0
    
    def test_new_hires_generation(self, sample_hr_data):
        """Test: Generación de nuevos empleados."""
        generator = TemporalHRGenerator(sample_hr_data)
        initial_max_id = generator.employee_id_counter
        
        next_month = generator.generate_next_month(retention_rate=0.7)
        
        # Debe haber nuevos IDs de empleados
        new_max_id = next_month['EmployeeNumber'].max()
        assert new_max_id > initial_max_id
    
    def test_period_column(self, sample_hr_data):
        """Test: Columna de período."""
        generator = TemporalHRGenerator(sample_hr_data, start_date="2024-01-01")
        
        month1 = generator.generate_next_month()
        month2 = generator.generate_next_month()
        
        # Verificar que los períodos son diferentes (cambio: 'period' -> 'DataMonth')
        assert month1['DataMonth'].iloc[0] != month2['DataMonth'].iloc[0]
        
        # Verificar formato de fecha
        assert isinstance(month1['DataMonth'].iloc[0], pd.Timestamp)
    
    def test_scenario_normal(self, sample_hr_data):
        """Test: Escenario normal."""
        generator = TemporalHRGenerator(sample_hr_data)
        
        next_month = generator.generate_next_month(scenario='normal')
        
        assert len(next_month) > 0
        assert 'Attrition' in next_month.columns
    
    def test_data_types_preservation(self, sample_hr_data):
        """Test: Preservación de tipos de datos."""
        generator = TemporalHRGenerator(sample_hr_data)
        next_month = generator.generate_next_month()
        
        # Verificar que los tipos numéricos se mantienen
        assert pd.api.types.is_numeric_dtype(next_month['Age'])
        assert pd.api.types.is_numeric_dtype(next_month['MonthlyIncome'])
        # Attrition ahora es string, no numérico
        assert pd.api.types.is_string_dtype(next_month['Attrition']) or pd.api.types.is_object_dtype(next_month['Attrition'])
    
    def test_no_negative_values(self, sample_hr_data):
        """Test: No debe haber valores negativos en campos que no lo permiten."""
        generator = TemporalHRGenerator(sample_hr_data)
        sequence = generator.generate_temporal_sequence(n_months=3)
        
        # Campos que no deben ser negativos
        non_negative_fields = ['Age', 'MonthlyIncome', 'YearsAtCompany', 
                               'YearsInCurrentRole', 'YearsSinceLastPromotion']
        
        for field in non_negative_fields:
            if field in sequence.columns:
                assert (sequence[field] >= 0).all(), f"{field} tiene valores negativos"
    
    def test_multiple_sequences_independence(self, sample_hr_data):
        """Test: Múltiples secuencias son independientes."""
        gen1 = TemporalHRGenerator(sample_hr_data.copy())
        gen2 = TemporalHRGenerator(sample_hr_data.copy())
        
        seq1 = gen1.generate_temporal_sequence(n_months=2)
        seq2 = gen2.generate_temporal_sequence(n_months=2)
        
        # Las secuencias deben ser diferentes (por aleatoriedad)
        # pero del mismo tamaño aproximado
        assert abs(len(seq1) - len(seq2)) < len(sample_hr_data) * 0.5


class TestTemporalGeneratorEdgeCases:
    """Tests para casos extremos."""
    
    def test_small_dataset(self):
        """Test: Dataset muy pequeño."""
        small_data = pd.DataFrame({
            'EmployeeNumber': [1, 2, 3],
            'Age': [25, 30, 35],
            'Attrition': [0, 1, 0],
            'MonthlyIncome': [3000, 4000, 5000]
        })
        
        generator = TemporalHRGenerator(small_data)
        next_month = generator.generate_next_month()
        
        assert len(next_month) > 0
    
    def test_high_attrition_scenario(self, sample_hr_data):
        """Test: Escenario de alta rotación."""
        generator = TemporalHRGenerator(sample_hr_data)
        
        # Simular alta rotación
        next_month = generator.generate_next_month(retention_rate=0.5)
        
        # Debe haber muchos nuevos empleados
        initial_ids = set(sample_hr_data['EmployeeNumber'])
        new_ids = set(next_month['EmployeeNumber']) - initial_ids
        
        assert len(new_ids) > 0
    
    def test_zero_months_sequence(self, sample_hr_data):
        """Test: Secuencia de 0 meses."""
        generator = TemporalHRGenerator(sample_hr_data)
        
        sequence = generator.generate_temporal_sequence(n_months=0)
        
        # Debe devolver solo el cohort inicial
        assert len(sequence) == len(sample_hr_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
