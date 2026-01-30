import requests
import pandas as pd
import json

try:
    df = pd.read_csv('data/data_v12.csv').head(2)
    data = df.to_dict('records')
    response = requests.post("http://localhost:8000/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Result: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
