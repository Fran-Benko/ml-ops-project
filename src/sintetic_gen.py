import pandas as pd
import numpy as np
import random
from src.utils import get_next_dataset_version
import os

def generate_hr_drift_dataset(n_per_batch=200):
    """
    Generates a synthetic HR dataset changing dynamically based on a random scenario.
    """
    np.random.seed(None) # Force randomness on each execution
    
    # Randomly select a Scenario
    scenarios = [
        {"name": "Normal Operation", "noise": 0.05, "drift_factor": 1.0, "description": "Baseline performance."},
        {"name": "The Great Resignation", "noise": 0.25, "drift_factor": 2.5, "description": "High turnover across all departments."},
        {"name": "Economic Recession", "noise": 0.15, "drift_factor": 1.8, "description": "Low raises and high distance stress."},
        {"name": "Toxic Culture Shift", "noise": 0.40, "drift_factor": 3.0, "description": "Drastic drop in satisfaction levels."},
        {"name": "Competitor Headhunting", "noise": 0.10, "drift_factor": 2.2, "description": "High-performers are leaving."}
    ]
    scenario = random.choice(scenarios)
    print(f"🎲 Generando datos bajo escenario: {scenario['name']} ({scenario['description']})")
    
    # --- 1. Configurations & Mappings ---
    mappings = {
        'Education': {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'},
        'EnvironmentSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'JobInvolvement': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'JobSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'PerformanceRating': {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'},
        'RelationshipSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'WorkLifeBalance': {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
    }
    
    vals = {
        'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
        'Department': ['Sales', 'Research & Development', 'Human Resources'],
        'EducationField': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'],
        'Gender': ['Female', 'Male'],
        'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 
                    'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'],
        'MaritalStatus': ['Single', 'Married', 'Divorced'],
        'OverTime': ['Yes', 'No']
    }

    batches = []
    
    # Simulating batches
    for b in range(1, 4):
        n = n_per_batch
        
        # --- Base Feature Generation ---
        data = {
            'Age': np.random.normal(37, 9, n).astype(int),
            'DailyRate': np.random.randint(100, 1500, n),
            'DistanceFromHome': np.random.randint(1, 30, n),
            'Education': np.random.randint(1, 6, n),
            'EnvironmentSatisfaction': np.random.randint(1, 5, n),
            'JobInvolvement': np.random.randint(1, 5, n),
            'JobLevel': np.random.randint(1, 6, n),
            'JobSatisfaction': np.random.randint(1, 5, n),
            'NumCompaniesWorked': np.random.randint(0, 10, n),
            'PercentSalaryHike': np.random.randint(11, 26, n),
            'PerformanceRating': np.random.choice([1, 2, 3, 4], n, p=[0.05, 0.05, 0.75, 0.15]),
            'RelationshipSatisfaction': np.random.randint(1, 5, n),
            'StandardHours': np.full(n, 80),
            'StockOptionLevel': np.random.randint(0, 4, n),
            'TotalWorkingYears': np.random.normal(11, 8, n).astype(int),
            'TrainingTimesLastYear': np.random.randint(0, 7, n),
            'WorkLifeBalance': np.random.randint(1, 5, n),
            'YearsAtCompany': np.random.normal(7, 6, n).astype(int),
            'EmployeeCount': np.full(n, 1),
            'EmployeeNumber': np.arange(1 + (b-1)*n, 1 + b*n),
            'Over18': np.full(n, 'Y'),
            'Batch_ID': np.full(n, b)
        }
        
        for k in ['Age', 'TotalWorkingYears', 'YearsAtCompany']:
            data[k] = np.clip(data[k], 0, 100)

        for c in vals:
            data[c] = np.random.choice(vals[c], n)
            
        # Income Gen with Noise
        inc = np.random.normal(6500, 4700, n)
        inc += np.random.normal(0, 2000 * scenario['noise'], n)
        data['MonthlyIncome'] = np.clip(inc, 2000, 20000).astype(int)

        # Drift Injection logic
        prob_attrition = np.full(n, 0.12 * scenario['drift_factor']) 

        if scenario['name'] == "Toxic Culture Shift":
            data['JobSatisfaction'] = np.random.choice([1, 2], n, p=[0.6, 0.4])
            data['EnvironmentSatisfaction'] = np.random.choice([1, 2], n, p=[0.7, 0.3])
        
        if scenario['name'] == "Economic Recession":
            data['PercentSalaryHike'] = np.random.randint(0, 5, n)
            data['MonthlyIncome'] = (data['MonthlyIncome'] * 0.8).astype(int)

        if scenario['name'] == "The Great Resignation":
            data['OverTime'] = np.random.choice(['Yes', 'No'], n, p=[0.8, 0.2])
            data['WorkLifeBalance'] = np.random.choice([1, 2], n, p=[0.5, 0.5])

        # Attrition Logic
        prob_attrition += np.where(data['JobSatisfaction'] == 1, 0.25, 0.0)
        prob_attrition += np.where(data['OverTime'] == 'Yes', 0.15, 0.0)
        prob_attrition += np.where(data['MonthlyIncome'] < 3500, 0.20, 0.0)
        
        if scenario['name'] == "Competitor Headhunting":
            prob_attrition += np.where(data['PerformanceRating'] >= 3, 0.40, 0.0)

        prob_attrition = np.clip(prob_attrition, 0, 0.95)
        data['Attrition'] = np.where(np.random.rand(n) < prob_attrition, 'Yes', 'No')
        
        batches.append(pd.DataFrame(data))
        
    df_final = pd.concat(batches, ignore_index=True)
    
    for col, mapping in mappings.items():
        if col in df_final.columns:
            df_final[col] = df_final[col].map(mapping)
            
    return df_final

if __name__ == "__main__":
    df_synthetic = generate_hr_drift_dataset(n_per_batch=400)
    print("Dataset Shape:", df_synthetic.shape)
    
    # Versioning
    if not os.path.exists('data'): os.makedirs('data')
    next_filename = get_next_dataset_version('data')
    save_path = f"data/{next_filename}"
    
    df_synthetic.to_csv(save_path, index=False)
    print(f"✅ Nueva versión de datos ({next_filename}) guardada.")
