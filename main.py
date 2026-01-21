from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Fraud Detection API")

# Load saved artifacts
model = joblib.load("fraud_model.pkl")
threshold = joblib.load("threshold.pkl")
columns = joblib.load("fraud_columns.pkl")

class Transaction(BaseModel):
    # We will send features as a dictionary
    features: dict

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(tx: Transaction):
    input_df = pd.DataFrame([tx.features])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    proba = float(model.predict_proba(input_df)[0][1])
    pred = int(proba >= threshold)

    return {
        "fraud_probability": proba,
        "fraud_prediction": pred,
        "threshold_used": float(threshold)
    }
