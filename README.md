# PlacementAI

**AI-Powered Placement Risk Modeling System for Indian Education Lenders**

PlacementAI predicts student employability risk 6 months before graduation and generates explainable, RBI-compliant risk intelligence for loan officers, portfolio managers, and relationship managers.

---

## Features

- **3-Layer AI Engine**: XGBoost placement predictor + Quantile Regression salary forecaster + SHAP explainability
- **10,000 Synthetic Student Records**: Logically correlated dataset with 52 features
- **FastAPI REST API**: Single/batch prediction, portfolio summary, fraud detection
- **Streamlit Dashboard**: 4-page lender-facing UI with interactive visualizations
- **RBI Compliant**: Risk intelligence only — no automated credit decisions

---

## Quick Start

### 1. Setup

```bash
# Clone and enter the project
cd placement-ai

# Create and activate venv
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python data/generate_dataset.py
```

### 3. Train Models

```bash
python -m model.train
python -m model.salary_model
```

### 4. Run API

```bash
uvicorn api.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 5. Run Dashboard

```bash
streamlit run app/dashboard.py
```

Dashboard: http://localhost:8501

---

## Docker Deployment

```bash
docker-compose up --build
```

- API: http://localhost:8000
- Dashboard: http://localhost:8501

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single student prediction |
| POST | `/predict/batch` | Batch prediction (CSV upload) |
| GET | `/portfolio/summary` | Portfolio-level statistics |
| GET | `/student/{id}/history` | Student risk score history |
| POST | `/fraud/flag` | Fraud anomaly detection |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "STU00001",
    "course_type": "Data Science/AI",
    "destination_country": "UK",
    "university_name": "University of Manchester",
    "institute_tier": 2,
    "cgpa": 7.8,
    "visa_type": "UK Graduate Route",
    "visa_risk_score": 2,
    "loan_amount_inr": 4800000,
    "graduation_year": 2025,
    "internship_count": 1,
    "internship_quality_tier": 2,
    "sector_hiring_trend": "Declining",
    "job_portal_activity_score": 4,
    "interview_pipeline_stage": 1,
    "loan_tenure_months": 120,
    "emi_amount_inr": 48000,
    "stem_flag": 1
  }'
```

---

## Model Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | >= 84% | 78.5% |
| F1 Macro | >= 79% | 75.5% |
| AUC-ROC | >= 0.91 | **0.918** |
| HIGH Recall | >= 79% | 73.4% |
| Salary MAPE | <= 11% | **7.5%** |

---

## Project Structure

```
placement-ai/
├── data/                    # Dataset generation
├── model/                   # ML pipeline (train, predict, SHAP, salary, fraud)
│   ├── artifacts/           # Trained model files (.pkl)
│   └── evaluation/          # Metrics + calibration plots
├── api/                     # FastAPI REST API
│   ├── routes/              # Endpoint handlers
│   └── tests/               # API tests
├── app/                     # Streamlit dashboard
│   └── pages/               # Dashboard pages (4)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | XGBoost, scikit-learn, SMOTE, SHAP |
| API | FastAPI, Pydantic v2, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Data | NumPy, Pandas, Faker |
| Deploy | Docker, docker-compose |

---

## Compliance

- RBI Fair Practices Code compliant
- No automated credit decisions
- SHAP explainability for every risk score
- Synthetic data only (no real PII)

---

*Built with PlacementAI*
