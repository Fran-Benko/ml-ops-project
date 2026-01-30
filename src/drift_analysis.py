import pandas as pd

def calculate_drift(df_old, df_new, column):
    """
    Calcula el drift estadístico básico entre dos versiones de un dataset.
    Para este MVP, usamos el cambio porcentual en la media para numéricas
    y el cambio en la distribución para categóricas.
    """
    if column not in df_old.columns or column not in df_new.columns:
        return None
    
    if pd.api.types.is_numeric_dtype(df_old[column]):
        mean_old = df_old[column].mean()
        mean_new = df_new[column].mean()
        drift = ((mean_new - mean_old) / mean_old) * 100 if mean_old != 0 else 0
        return {"type": "numeric", "mean_old": mean_old, "mean_new": mean_new, "drift_pct": drift}
    else:
        dist_old = df_old[column].value_counts(normalize=True).to_dict()
        dist_new = df_new[column].value_counts(normalize=True).to_dict()
        return {"type": "categorical", "dist_old": dist_old, "dist_new": dist_new}

def compare_datasets(path_old, path_new):
    """
    Compara dos datasets y devuelve un resumen de drift.
    """
    df_old = pd.read_csv(path_old)
    df_new = pd.read_csv(path_new)
    
    # Columnas clave para monitorear drift (KPIs de RRHH)
    kpis = ['MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance', 'DistanceFromHome']
    
    results = {}
    for col in kpis:
        results[col] = calculate_drift(df_old, df_new, col)
        
    return results
