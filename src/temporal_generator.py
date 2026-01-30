"""
Generador Temporal de Datos de RRHH con Continuidad de Cohortes
================================================================

Este módulo implementa un generador de datos sintéticos que mantiene
continuidad temporal entre períodos, simulando la evolución real de
una fuerza laboral donde los empleados persisten mes a mes.

Características:
- Continuidad de empleados entre períodos (80-90% retención)
- Envejecimiento natural de cohortes
- Drift gradual y realista
- Generación de nuevos empleados para reemplazos
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


class TemporalHRGenerator:
    """
    Generador de datos de RRHH con continuidad temporal.
    
    Mantiene una cohorte de empleados que evoluciona mes a mes,
    simulando attrition real, envejecimiento y drift gradual.
    """
    
    def __init__(self, seed_data: pd.DataFrame, start_date: str = "2024-01-01"):
        """
        Inicializa el generador con datos semilla.
        
        Args:
            seed_data: DataFrame con datos iniciales de empleados
            start_date: Fecha de inicio de la simulación (formato YYYY-MM-DD)
        """
        self.current_cohort = seed_data.copy()
        self.start_date = pd.to_datetime(start_date)
        self.current_date = self.start_date
        self.month_counter = 0
        
        # Asegurar que tenemos EmployeeNumber
        if 'EmployeeNumber' not in self.current_cohort.columns:
            self.current_cohort['EmployeeNumber'] = range(1, len(self.current_cohort) + 1)
        
        self.employee_id_counter = self.current_cohort['EmployeeNumber'].max()
        
        # Agregar columna de fecha si no existe
        if 'DataMonth' not in self.current_cohort.columns:
            self.current_cohort['DataMonth'] = self.current_date
        
        # Convertir Attrition a string si es necesario
        if 'Attrition' in self.current_cohort.columns:
            self.current_cohort['Attrition'] = self.current_cohort['Attrition'].astype(str)
        
        # Configuraciones de generación
        self.categorical_mappings = {
            'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
            'Department': ['Sales', 'Research & Development', 'Human Resources'],
            'EducationField': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 
                              'Human Resources', 'Other'],
            'Gender': ['Female', 'Male'],
            'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 
                       'Manufacturing Director', 'Healthcare Representative', 'Manager', 
                       'Sales Representative', 'Research Director', 'Human Resources'],
            'MaritalStatus': ['Single', 'Married', 'Divorced'],
            'OverTime': ['Yes', 'No']
        }
        
        self.ordinal_mappings = {
            'Education': ['Below College', 'College', 'Bachelor', 'Master', 'Doctor'],
            'EnvironmentSatisfaction': ['Low', 'Medium', 'High', 'Very High'],
            'JobInvolvement': ['Low', 'Medium', 'High', 'Very High'],
            'JobSatisfaction': ['Low', 'Medium', 'High', 'Very High'],
            'PerformanceRating': ['Low', 'Good', 'Excellent', 'Outstanding'],
            'RelationshipSatisfaction': ['Low', 'Medium', 'High', 'Very High'],
            'WorkLifeBalance': ['Bad', 'Good', 'Better', 'Best']
        }
        
        print(f"✅ TemporalHRGenerator inicializado con {len(self.current_cohort)} empleados")
        print(f"📅 Fecha inicial: {self.start_date.strftime('%Y-%m-%d')}")
    
    def generate_next_month(self, 
                           retention_rate: float = 0.85,
                           drift_params: Optional[Dict] = None,
                           scenario: str = "normal") -> pd.DataFrame:
        """
        Genera datos para el siguiente mes manteniendo continuidad.
        
        Args:
            retention_rate: Proporción de empleados que permanecen (0.0-1.0)
            drift_params: Parámetros de drift gradual (opcional)
            scenario: Escenario de negocio ('normal', 'recession', 'growth')
        
        Returns:
            DataFrame con datos del nuevo mes
        """
        self.month_counter += 1
        self.current_date = self.start_date + timedelta(days=30 * self.month_counter)
        
        print(f"\n🔄 Generando Mes {self.month_counter} ({self.current_date.strftime('%Y-%m')})")
        
        # 1. Simular attrition natural
        retained_employees = self._simulate_attrition(retention_rate, scenario)
        print(f"   👥 Empleados retenidos: {len(retained_employees)}")
        
        # 2. Envejecer cohorte
        aged_employees = self._age_cohort(retained_employees)
        
        # 3. Aplicar drift gradual
        if drift_params:
            aged_employees = self._apply_gradual_drift(aged_employees, drift_params)
        
        # 4. Generar nuevos empleados (reemplazos)
        n_new = len(self.current_cohort) - len(aged_employees)
        new_employees = self._generate_new_hires(n_new, scenario)
        print(f"   🆕 Nuevos empleados: {n_new}")
        
        # 5. Combinar cohortes
        self.current_cohort = pd.concat([aged_employees, new_employees], ignore_index=True)
        self.current_cohort['DataMonth'] = self.current_date
        
        # 6. Recalcular attrition para el próximo período
        self.current_cohort = self._recalculate_attrition_risk(self.current_cohort, scenario)
        
        print(f"   ✅ Total empleados: {len(self.current_cohort)}")
        attrition_rate = (self.current_cohort['Attrition'] == 'Yes').mean()
        print(f"   📊 Tasa de attrition: {attrition_rate:.2%}")
        
        return self.current_cohort.copy()
    
    def _simulate_attrition(self, retention_rate: float, scenario: str) -> pd.DataFrame:
        """Simula attrition basado en características de empleados."""
        # Filtrar empleados que NO se fueron (Attrition == 'No')
        active_employees = self.current_cohort[
            self.current_cohort['Attrition'].isin(['No', '0', 0])
        ].copy()
        
        if len(active_employees) == 0:
            return pd.DataFrame()
        
        # Ajustar retention_rate según escenario
        if scenario == "recession":
            retention_rate = min(0.95, retention_rate + 0.05)  # Menos gente se va
        elif scenario == "growth":
            retention_rate = max(0.75, retention_rate - 0.05)  # Más competencia
        
        # Retención con algo de aleatoriedad
        n_retained = int(len(active_employees) * retention_rate)
        retained = active_employees.sample(n=n_retained, random_state=None)
        
        return retained
    
    def _age_cohort(self, employees: pd.DataFrame) -> pd.DataFrame:
        """Envejece la cohorte: incrementa edad y años en la empresa."""
        aged = employees.copy()
        
        # Incrementar edad (1 mes ≈ 0.08 años, redondeamos cada 12 meses)
        if self.month_counter % 12 == 0:
            aged['Age'] = aged['Age'] + 1
        
        # Incrementar años en la empresa (más granular)
        if 'YearsAtCompany' in aged.columns:
            aged['YearsAtCompany'] = aged['YearsAtCompany'] + (1/12)
            aged['YearsAtCompany'] = aged['YearsAtCompany'].round(1)
        
        # Incrementar años totales de trabajo
        if 'TotalWorkingYears' in aged.columns:
            aged['TotalWorkingYears'] = aged['TotalWorkingYears'] + (1/12)
            aged['TotalWorkingYears'] = aged['TotalWorkingYears'].round(1)
        
        # Incrementar años en rol actual (con límite basado en YearsAtCompany)
        if 'YearsInCurrentRole' in aged.columns:
            aged['YearsInCurrentRole'] = aged['YearsInCurrentRole'] + (1/12)
            aged['YearsInCurrentRole'] = aged['YearsInCurrentRole'].round(1)
            # Asegurar que no exceda YearsAtCompany
            if 'YearsAtCompany' in aged.columns:
                aged['YearsInCurrentRole'] = aged[['YearsInCurrentRole', 'YearsAtCompany']].min(axis=1)
            # Asegurar que no sea negativo
            aged['YearsInCurrentRole'] = aged['YearsInCurrentRole'].clip(lower=0)
        
        # Incrementar años desde última promoción
        if 'YearsSinceLastPromotion' in aged.columns:
            aged['YearsSinceLastPromotion'] = aged['YearsSinceLastPromotion'] + (1/12)
            aged['YearsSinceLastPromotion'] = aged['YearsSinceLastPromotion'].round(1)
            # Asegurar que no exceda YearsAtCompany
            if 'YearsAtCompany' in aged.columns:
                aged['YearsSinceLastPromotion'] = aged[['YearsSinceLastPromotion', 'YearsAtCompany']].min(axis=1)
            # Asegurar que no sea negativo
            aged['YearsSinceLastPromotion'] = aged['YearsSinceLastPromotion'].clip(lower=0)
        
        # Incrementar años con manager actual
        if 'YearsWithCurrManager' in aged.columns:
            aged['YearsWithCurrManager'] = aged['YearsWithCurrManager'] + (1/12)
            aged['YearsWithCurrManager'] = aged['YearsWithCurrManager'].round(1)
            # Asegurar que no exceda YearsAtCompany
            if 'YearsAtCompany' in aged.columns:
                aged['YearsWithCurrManager'] = aged[['YearsWithCurrManager', 'YearsAtCompany']].min(axis=1)
            # Asegurar que no sea negativo
            aged['YearsWithCurrManager'] = aged['YearsWithCurrManager'].clip(lower=0)
        
        return aged
    
    def _apply_gradual_drift(self, employees: pd.DataFrame, 
                            drift_params: Dict) -> pd.DataFrame:
        """
        Aplica drift gradual a las características de empleados.
        
        Args:
            drift_params: Dict con parámetros como:
                - 'salary_increase': % de incremento salarial
                - 'satisfaction_decay': Reducción en satisfacción
                - 'overtime_increase': Incremento en horas extra
        """
        drifted = employees.copy()
        
        # Drift en salario (incremento gradual)
        if 'salary_increase' in drift_params and 'MonthlyIncome' in drifted.columns:
            increase = drift_params['salary_increase']
            drifted['MonthlyIncome'] = (drifted['MonthlyIncome'] * (1 + increase)).astype(int)
        
        # Drift en satisfacción (puede decrecer)
        if 'satisfaction_decay' in drift_params and 'JobSatisfaction' in drifted.columns:
            decay = drift_params['satisfaction_decay']
            # Aplicar decay probabilístico
            mask = np.random.rand(len(drifted)) < abs(decay)
            if decay < 0:  # Decremento
                current_satisfaction = drifted.loc[mask, 'JobSatisfaction']
                satisfaction_map = {'Very High': 'High', 'High': 'Medium', 
                                   'Medium': 'Low', 'Low': 'Low'}
                drifted.loc[mask, 'JobSatisfaction'] = current_satisfaction.map(
                    lambda x: satisfaction_map.get(x, x)
                )
        
        # Drift en overtime
        if 'overtime_increase' in drift_params and 'OverTime' in drifted.columns:
            increase = drift_params['overtime_increase']
            mask = np.random.rand(len(drifted)) < increase
            drifted.loc[mask, 'OverTime'] = 'Yes'
        
        return drifted
    
    def _generate_new_hires(self, n_new: int, scenario: str) -> pd.DataFrame:
        """Genera nuevos empleados (reemplazos)."""
        if n_new <= 0:
            return pd.DataFrame()
        
        new_employees = {}
        
        # IDs únicos
        new_ids = range(self.employee_id_counter + 1, 
                       self.employee_id_counter + n_new + 1)
        new_employees['EmployeeNumber'] = list(new_ids)
        self.employee_id_counter += n_new
        
        # Características demográficas
        new_employees['Age'] = np.random.normal(30, 7, n_new).astype(int)
        new_employees['Age'] = np.clip(new_employees['Age'], 22, 65)
        
        # Características laborales (nuevos empleados típicamente junior)
        new_employees['YearsAtCompany'] = np.random.uniform(0, 0.5, n_new).round(1)
        new_employees['TotalWorkingYears'] = np.random.normal(5, 4, n_new).astype(int)
        new_employees['TotalWorkingYears'] = np.clip(new_employees['TotalWorkingYears'], 0, 40)
        
        # Salario (nuevos empleados ganan menos)
        base_salary = 4000 if scenario == "recession" else 5000
        new_employees['MonthlyIncome'] = np.random.normal(base_salary, 1500, n_new).astype(int)
        new_employees['MonthlyIncome'] = np.clip(new_employees['MonthlyIncome'], 2500, 15000)
        
        # Otras características numéricas
        new_employees['DailyRate'] = np.random.randint(100, 1500, n_new)
        new_employees['DistanceFromHome'] = np.random.randint(1, 30, n_new)
        new_employees['JobLevel'] = np.random.choice([1, 2, 3], n_new, p=[0.6, 0.3, 0.1])
        new_employees['NumCompaniesWorked'] = np.random.randint(0, 5, n_new)
        new_employees['PercentSalaryHike'] = np.random.randint(11, 20, n_new)
        new_employees['StockOptionLevel'] = np.random.randint(0, 3, n_new)
        new_employees['TrainingTimesLastYear'] = np.random.randint(0, 5, n_new)
        
        # Campos relacionados con tiempo en la empresa (nuevos empleados)
        new_employees['YearsInCurrentRole'] = np.random.uniform(0, 0.5, n_new).round(1)
        new_employees['YearsSinceLastPromotion'] = np.zeros(n_new)  # Recién contratados
        new_employees['YearsWithCurrManager'] = np.random.uniform(0, 0.5, n_new).round(1)
        
        # Características categóricas
        for col, values in self.categorical_mappings.items():
            new_employees[col] = np.random.choice(values, n_new)
        
        # Características ordinales
        for col, values in self.ordinal_mappings.items():
            new_employees[col] = np.random.choice(values, n_new)
        
        # Constantes
        new_employees['EmployeeCount'] = 1
        new_employees['Over18'] = 'Y'
        new_employees['StandardHours'] = 80
        
        # Attrition inicial (nuevos empleados tienen mayor riesgo)
        new_employees['Attrition'] = np.random.choice(
            ['Yes', 'No'], n_new, p=[0.20, 0.80]
        )
        
        df_new = pd.DataFrame(new_employees)
        return df_new
    
    def _recalculate_attrition_risk(self, employees: pd.DataFrame, 
                                   scenario: str) -> pd.DataFrame:
        """Recalcula el riesgo de attrition para el próximo período."""
        df = employees.copy()
        
        # Probabilidad base según escenario
        base_prob = {
            'normal': 0.12,
            'recession': 0.08,
            'growth': 0.18
        }.get(scenario, 0.12)
        
        prob_attrition = np.full(len(df), base_prob)
        
        # Factores de riesgo
        if 'JobSatisfaction' in df.columns:
            prob_attrition += np.where(df['JobSatisfaction'] == 'Low', 0.25, 0.0)
        
        if 'OverTime' in df.columns:
            prob_attrition += np.where(df['OverTime'] == 'Yes', 0.15, 0.0)
        
        if 'MonthlyIncome' in df.columns:
            prob_attrition += np.where(df['MonthlyIncome'] < 3500, 0.20, 0.0)
        
        if 'YearsAtCompany' in df.columns:
            prob_attrition += np.where(df['YearsAtCompany'] < 1, 0.15, 0.0)
        
        # Clip probabilidades
        prob_attrition = np.clip(prob_attrition, 0, 0.95)
        
        # Asignar attrition
        df['Attrition'] = np.where(
            np.random.rand(len(df)) < prob_attrition, 'Yes', 'No'
        )
        
        return df
    
    def generate_temporal_sequence(self, 
                                  n_months: int = 6,
                                  retention_rate: float = 0.85,
                                  drift_schedule: Optional[List[Dict]] = None,
                                  scenario_schedule: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Genera una secuencia temporal de múltiples meses.
        
        Args:
            n_months: Número de meses a generar
            retention_rate: Tasa de retención base
            drift_schedule: Lista de drift_params por mes (opcional)
            scenario_schedule: Lista de escenarios por mes (opcional)
        
        Returns:
            DataFrame concatenado con todos los meses
        """
        all_months = [self.current_cohort.copy()]
        
        for month in range(n_months):
            drift_params = drift_schedule[month] if drift_schedule else None
            scenario = scenario_schedule[month] if scenario_schedule else "normal"
            
            next_month = self.generate_next_month(
                retention_rate=retention_rate,
                drift_params=drift_params,
                scenario=scenario
            )
            all_months.append(next_month.copy())
        
        # Concatenar todos los meses
        full_sequence = pd.concat(all_months, ignore_index=True)
        
        print(f"\n✅ Secuencia temporal generada: {n_months + 1} meses, {len(full_sequence)} registros")
        
        return full_sequence


if __name__ == "__main__":
    # Ejemplo de uso
    print("=" * 70)
    print("DEMO: TemporalHRGenerator")
    print("=" * 70)
    
    # Cargar datos semilla
    seed_data = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    print(f"\n📊 Datos semilla cargados: {seed_data.shape}")
    
    # Inicializar generador
    generator = TemporalHRGenerator(seed_data, start_date="2024-01-01")
    
    # Generar secuencia de 6 meses con drift gradual
    drift_schedule = [
        None,  # Mes 1: sin drift
        {'salary_increase': 0.02},  # Mes 2: +2% salario
        {'salary_increase': 0.02, 'satisfaction_decay': -0.1},  # Mes 3
        {'satisfaction_decay': -0.15, 'overtime_increase': 0.1},  # Mes 4
        {'overtime_increase': 0.15},  # Mes 5
        {'satisfaction_decay': -0.20},  # Mes 6
    ]
    
    scenario_schedule = ['normal', 'normal', 'normal', 'growth', 'growth', 'recession']
    
    temporal_data = generator.generate_temporal_sequence(
        n_months=6,
        retention_rate=0.85,
        drift_schedule=drift_schedule,
        scenario_schedule=scenario_schedule
    )
    
    # Guardar resultado
    output_path = 'data/temporal_sequence_demo.csv'
    temporal_data.to_csv(output_path, index=False)
    print(f"\n💾 Datos guardados en: {output_path}")
    
    # Estadísticas por mes
    print("\n📈 Estadísticas por mes:")
    stats = temporal_data.groupby('DataMonth').agg({
        'EmployeeNumber': 'count',
        'Attrition': lambda x: (x == 'Yes').mean(),
        'MonthlyIncome': 'mean',
        'JobSatisfaction': lambda x: (x == 'Low').mean()
    }).round(3)
    stats.columns = ['N_Empleados', 'Tasa_Attrition', 'Salario_Promedio', 'Pct_Baja_Satisfaccion']
    print(stats)

# Made with Bob
