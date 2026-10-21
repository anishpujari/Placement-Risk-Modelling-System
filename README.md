# PlacementAI

**AI-Powered Placement Risk Modeling System for Indian Education Lenders**

PlacementAI predicts student employability risk 6 months before graduation and generates explainable, RBI-compliant risk intelligence for loan officers, portfolio managers, and relationship managers.

---

## Features

- **3-Layer AI Engine**: XGBoost placement predictor + Quantile Regression salary forecaster + SHAP explainability
- **100,000 Synthetic Student Records**: Logically correlated dataset with 52 features
- **FastAPI REST API**: Single/batch prediction, portfolio summary, fraud detection
- **Streamlit Dashboard**: 5-page lender-facing UI (portfolio, institutes, student lookup, new assessment, model insights)
- **Detailed SHAP**: Per-student contribution of all 37 features, portfolio-level importance and per-feature dependence plots
- **PDF risk cards**: One-page printable card per student or applicant, with SHAP drivers
- **Cost-sensitive decision rule**: HIGH-risk threshold tuned on validation data so fewer risky borrowers are missed
- **RBI Compliant**: Risk intelligence only, no automated credit decisions

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
python -m model.compare_models   # optional: baseline comparison shown on Model insights
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
    "student_id": "STU000001",
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

Evaluated on the held-out 2025 graduating cohort (34,759 students). Training uses the 2023 and 2024 cohorts only.

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | >= 84% | 78.0% (ceiling 79.7%, see below) |
| F1 Macro | >= 79% | 74.5% |
| AUC-ROC | >= 0.91 | **0.923** |
| HIGH Recall | >= 79% | **80.3%** |
| HIGH Precision | - | 73.9% |
| Salary MAPE | <= 11% | **7.6%** |

**HIGH-risk threshold.** A student is rated HIGH whenever P(HIGH) >= 0.34, otherwise the more likely of MEDIUM and LOW. The threshold was chosen on a 20% validation slice of the training years (never the test set) as the highest value that keeps HIGH recall at or above 80%. Missing a high-risk borrower costs a lender more than reviewing an extra file. With plain argmax the model scores 78.7% accuracy but only 73.7% HIGH recall.

**Accuracy ceiling.** Each risk tier is derived from a placement score plus random noise (standard deviation about 8.2 points, see `data/generate_dataset.py`). An oracle that knows the exact scoring formula, but not the noise, reaches only **79.7%** accuracy on the test cohort (`model/ceiling.py`). The 84% accuracy target is therefore not reachable on this data by any model. The classifier is within 1 point of the ceiling.

**Baselines.** Same features, training years and SMOTE balancing, scored on the 2025 cohort (`python -m model.compare_models`):

| Model | Accuracy | F1 Macro | AUC-ROC | HIGH Recall |
|-------|----------|----------|---------|-------------|
| Always predict LOW | 55.9% | 23.9% | 0.500 | 0.0% |
| Decision tree | 74.9% | 71.8% | 0.895 | 64.5% |
| Logistic regression | 77.5% | 73.4% | 0.917 | 73.5% |
| Random forest | 77.8% | 74.6% | 0.914 | 70.3% |
| XGBoost | **78.7%** | **75.6%** | **0.923** | 73.7% |
| XGBoost, tuned for HIGH risk | 78.0% | 74.5% | **0.923** | **80.3%** |

Full metrics, including per-class results and the ceiling report, are written to `model/evaluation/metrics_report.json`. The dashboard's **Model insights** page shows them with the confusion matrix, threshold trade-off, calibration and SHAP importance.

--------|--------|----------|
| Accuracy | >= 84% | 78.7% |
| F1 Macro | >= 79% | 75.6% |
| AUC-ROC | >= 0.91 | **0.923** |
| HIGH Recall | >= 79% | 73.7% |
| Salary MAPE | <= 11% | **7.6%** |

---

## Project Structure

```
placement-ai/
├── data/                    # Dataset generation
├── model/                   # ML pipeline (train, predict, SHAP, salary, fraud, ceiling)
│   ├── artifacts/           # Trained model files (.pkl)
│   └── evaluation/          # Metrics + calibration plots
├── api/                     # FastAPI REST API
│   ├── routes/              # Endpoint handlers
│   └── tests/               # API tests
├── app/                     # Streamlit dashboard
│   ├── ui.py                # Shared theme, charts and SHAP views
│   └── views/               # Dashboard pages (5)
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
