from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import mlflow
from typing import List

app = FastAPI(title="Employee Attrition Serving API")

# Setup MLflow
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)

MODEL_PATH = "models/churn_pipeline.joblib"

class EmployeeData(BaseModel):
    Age: int
    DailyRate: int
    DistanceFromHome: int
    Education: str
    EnvironmentSatisfaction: str
    JobInvolvement: str
    JobLevel: int
    JobSatisfaction: str
    NumCompaniesWorked: int
    PercentSalaryHike: int
    PerformanceRating: str
    RelationshipSatisfaction: str
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: str
    YearsAtCompany: int
    MonthlyIncome: int
    BusinessTravel: str
    Department: str
    EducationField: str
    Gender: str
    JobRole: str
    MaritalStatus: str
    OverTime: str
    # Batch_ID is optional - not used by the model (dropped during training)
    Batch_ID: int = None
    # Add constant columns that might be missing in some older samples but needed by the model
    EmployeeCount: int = 1
    EmployeeNumber: int = 0
    Over18: str = 'Y'
    StandardHours: int = 80

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: List[EmployeeData]):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")
    
    try:
        # Load model
        model = joblib.load(MODEL_PATH)
        
        # Convert input to DataFrame
        df = pd.DataFrame([item.dict() for item in data])
        
        # Drop columns that were removed during training
        # These columns are not used by the model
        drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber', 'Batch_ID']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # --- Feature Alignment via Model Introspection ---
        # Instead of manual drops, we ask the model exactly what it wants.
        # Transformers list: [('num', Pipe, num_cols), ('cat', Pipe, cat_cols)]
        num_features = model.named_steps['preprocessor'].transformers_[0][2]
        cat_features = model.named_steps['preprocessor'].transformers_[1][2]
        required_features = num_features + cat_features
        
        # Ensure all required features are present in the DF
        # We fill missing with 0 or empty values as a safety net
        for f in required_features:
            if f not in df.columns:
                df[f] = 0 if f in num_features else "Unknown"

        # Reorder DataFrame to match the exact order seen during fit()
        df_final = df[required_features]

        print(f"DEBUG: Processing {len(df_final.columns)} columns in exact order.")
        
        # Generate predictions
        preds_class = model.predict(df_final)
        preds_proba = model.predict_proba(df_final)[:, 1]
        
        results = []
        for i in range(len(df)):
            results.append({
                "prediction": int(preds_class[i]),
                "probability": float(preds_proba[i])
            })
            
        return results
    except Exception as e:
        print(f"PREDICTION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
