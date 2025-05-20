
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List
import openai

app = FastAPI()

# Load model and scaler
MODEL_PATH = "model/diabetes_model.pkl"
SCALER_PATH = "model/diabetes_scaler.pkl"
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# Input data model
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict_risk(data: PatientData):
    try:
        # Format input
        features = np.array([[
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.SkinThickness,
            data.Insulin,
            data.BMI,
            data.DiabetesPedigreeFunction,
            data.Age
        ]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        return {"prediction": int(prediction), "probability": round(probability, 3)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain")
def explain_prediction(data: PatientData):
    try:
        explanation = f"""
        This patient's risk is assessed based on:
        - Glucose level: {data.Glucose}
        - BMI: {data.BMI}
        - Insulin level: {data.Insulin}
        - Age: {data.Age}
        These features are commonly associated with diabetes risk in clinical research.
        """
        return {"explanation": explanation.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
