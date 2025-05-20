# 🧠 MedAI Explain – Backend (FastAPI)

This is the backend API for **MedAI Explain**, a Type 2 Diabetes risk prediction and explainability tool.

## 🔍 What It Does
- Predicts diabetes risk from 8 clinical features using a trained Random Forest Classifier
- Returns a risk score and probability
- Provides a plain English explanation of contributing factors
- Exposes REST API endpoints via FastAPI

## 📦 Stack
- Python, FastAPI
- scikit-learn, joblib
- Hosted on Render

## 🚀 API Endpoints
### `POST /predict`
Takes patient data JSON, returns prediction (0/1) and probability.

### `POST /explain`
Returns a GPT-like natural language explanation of the prediction logic.

### `GET /rmse`
(Optional) Returns model RMSE using internal test data.

## 🛠 Setup
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## 📁 Files
- `main.py`: FastAPI application
- `diabetes_model.pkl`: Trained Random Forest model
- `diabetes_scaler.pkl`: StandardScaler for input normalization

## 🧠 Example Input
```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 80,
  "BMI": 28.0,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 35
}
```
