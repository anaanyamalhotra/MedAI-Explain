# 🧠 MedAI Explain – LLM-Powered Diabetes Risk Explainer

**Live Demo:** https://medai-explain-frontend.streamlit.app/
**Author:** Ananya Malhotra

---

## 📊 Overview

MedAI Explain predicts the risk of Type 2 Diabetes using key medical metrics (e.g., glucose, insulin, BMI), and explains the results using natural language powered by LLMs.

---

## 🚀 API Endpoints (FastAPI)

### `POST /predict`
Predicts diabetes risk from 8 clinical features.

**Input JSON:**
```json
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 80,
  "SkinThickness": 25,
  "Insulin": 94,
  "BMI": 28.2,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 35
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.842
}
```

---

### `POST /explain`
Returns a natural-language explanation of the input's risk factors.

**Response:**
```json
{
  "explanation": "This patient's risk is assessed based on: ..."
}
```

---

## 📦 Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 🧠 Model

- Random Forest Classifier trained on Pima Indians Diabetes dataset
- Accuracy ~75%
- Preprocessing: Median imputation + StandardScaler

---

## 📁 Structure

```
backend/
├── main.py
├── model/
│   ├── diabetes_model.pkl
│   └── diabetes_scaler.pkl
└── requirements.txt
```
