"""
PlacementAI - API Tests
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "PlacementAI"


def test_predict_single():
    payload = {
        "student_id": "TEST001",
        "course_type": "Engineering/CS",
        "destination_country": "Canada",
        "university_name": "University of Toronto",
        "institute_tier": 1,
        "cgpa": 8.5,
        "internship_count": 2,
        "internship_quality_tier": 3,
        "visa_type": "PGWP",
        "visa_risk_score": 1,
        "city_name": "Toronto",
        "city_tier": 1,
        "region_job_density": 4,
        "target_sector": "Technology",
        "sector_hiring_trend": "Growing",
        "macro_condition_index": 3.8,
        "job_portal_activity_score": 4,
        "interview_pipeline_stage": 2,
        "resume_updated_flag": 1,
        "skillup_event_flag": 1,
        "loan_amount_inr": 3500000,
        "loan_tenure_months": 120,
        "emi_amount_inr": 35000,
        "graduation_year": 2025,
        "stem_flag": 1,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["student_id"] == "TEST001"
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert "placement_probabilities" in data
    assert len(data["shap_drivers"]) == 3
    # The route falls back to a placeholder if SHAP fails; make sure it didn't
    assert all("unavailable" not in d["feature"].lower() for d in data["shap_drivers"])
    assert all(d["impact"][0] in "+-" for d in data["shap_drivers"])


def test_portfolio_summary():
    r = client.get("/portfolio/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["total_active_accounts"] > 0
    assert "risk_distribution" in data


def test_student_history():
    r = client.get("/student/STU000001/history")
    assert r.status_code == 200
    data = r.json()
    assert data["student_id"] == "STU000001"
    assert len(data["history"]) > 0


def test_student_not_found():
    r = client.get("/student/NONEXISTENT/history")
    assert r.status_code == 404


def test_fraud_flag():
    r = client.post("/fraud/flag")
    assert r.status_code == 200
    data = r.json()
    assert data["total_checked"] > 0
