"""
PlacementAI - Predict Routes
POST /predict and POST /predict/batch
"""

import os, io, json
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException

from api.schemas import (
    StudentPredictRequest, StudentPredictResponse,
    PlacementProbabilities, SalaryForecast, ShapDriver,
)
from model.predict import predict_student, load_model, prepare_features
from model.shap_explainer import get_explainer, explain_student, generate_shap_summary
from model.salary_model import predict_salary

router = APIRouter()

# Lazy-loaded globals
_model = None
_encoders = None
_feature_cols = None
_explainer = None

ACTION_MAP = {
    "Technology": ["AWS/Azure certification", "Build GitHub portfolio", "Mock technical interviews"],
    "Banking/Finance": ["CFA Level 1 prep", "Bloomberg terminal certification", "LinkedIn Finance group networking"],
    "Consulting/Management": ["Case interview coaching", "MBA career fair participation", "MBA alumni outreach"],
    "Healthcare": ["Local licensing exam prep", "Hospital volunteer program", "Nursing agency registration"],
}


def _ensure_loaded():
    global _model, _encoders, _feature_cols, _explainer
    if _model is None:
        _model, _encoders, _feature_cols = load_model()
        _explainer = get_explainer(_model)


def _get_actions(risk_tier, sector, shap_drivers, loan_amount):
    actions = list(ACTION_MAP.get(sector, ["Review career plan"]))
    if risk_tier == "HIGH" and loan_amount > 4000000:
        actions.append("Flag for EMI holiday discussion")
    if any("resume" in str(d).lower() for d in shap_drivers):
        actions.append("Resume gap: add project portfolio")
    return actions[:4]


def _full_predict(student_dict):
    """Full prediction pipeline for a single student."""
    _ensure_loaded()

    # Risk prediction
    result = predict_student(student_dict, _model, _encoders, _feature_cols)

    # SHAP explanation
    X = prepare_features(student_dict, _encoders, _feature_cols)
    try:
        drivers = explain_student(X, _explainer, _feature_cols, student_dict)
    except Exception:
        drivers = [{"feature": "Model explanation unavailable", "impact": "N/A", "raw_shap": 0}]

    shap_drivers = [ShapDriver(feature=d["feature"], impact=d["impact"]) for d in drivers]
    summary = generate_shap_summary(drivers)

    # Salary forecast
    try:
        salary = predict_salary(student_dict)
        salary_forecast = SalaryForecast(**salary)
    except Exception:
        salary_forecast = None

    # Actions
    sector = student_dict.get("target_sector", "Technology")
    loan = student_dict.get("loan_amount_inr", 0)
    actions = _get_actions(result["risk_tier"], sector, drivers, loan)

    emi_flag = result["risk_tier"] == "HIGH" and loan > 4000000

    return StudentPredictResponse(
        student_id=student_dict.get("student_id", "UNKNOWN"),
        risk_tier=result["risk_tier"],
        risk_score=result["risk_score"],
        placement_probabilities=PlacementProbabilities(**result["placement_probabilities"]),
        salary_forecast=salary_forecast,
        shap_drivers=shap_drivers,
        ai_risk_summary=summary,
        recommended_actions=actions,
        next_rescore_days=30,
        emi_holiday_flag=emi_flag,
    )


@router.post("/predict", response_model=StudentPredictResponse)
async def predict_single(request: StudentPredictRequest):
    """Single student prediction with full risk card output."""
    student_dict = request.model_dump()
    return _full_predict(student_dict)


@router.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction from CSV upload."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    results = []
    for _, row in df.iterrows():
        student_dict = row.to_dict()
        try:
            pred = _full_predict(student_dict)
            results.append(pred.model_dump())
        except Exception as e:
            results.append({"student_id": student_dict.get("student_id", "?"), "error": str(e)})

    # Portfolio summary
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for r in results:
        tier = r.get("risk_tier", "")
        if tier in risk_counts:
            risk_counts[tier] += 1

    return {
        "predictions": results,
        "summary": {
            "total": len(results),
            "risk_distribution": risk_counts,
        }
    }
