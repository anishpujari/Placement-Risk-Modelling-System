"""
PlacementAI - Portfolio Routes
GET /portfolio/summary and GET /student/{student_id}/history
"""

import os
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from api.schemas import PortfolioSummary

router = APIRouter()

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "placement_10k_final.csv")


def _load_data():
    return pd.read_csv(DATA_PATH)


@router.get("/portfolio/summary", response_model=PortfolioSummary)
async def portfolio_summary():
    """Aggregate portfolio statistics."""
    df = _load_data()

    total = len(df)
    high_risk = int((df["risk_tier"] == "HIGH").sum())

    # Students graduating within 90 days (approximate from 2025 cohort)
    grad_90 = int((df["graduation_year"] == 2025).sum() // 3)

    at_risk_loan = df[df["risk_tier"] == "HIGH"]["loan_amount_inr"].sum() / 1e7  # in Cr

    risk_dist = df["risk_tier"].value_counts().to_dict()
    country_bd = df.groupby("destination_country")["risk_tier"].value_counts().unstack(fill_value=0).to_dict(orient="index")

    # Sector heatmap: sector × risk tier counts
    sector_hm = df.groupby("target_sector")["risk_tier"].value_counts().unstack(fill_value=0).to_dict(orient="index")

    return PortfolioSummary(
        total_active_accounts=total,
        high_risk_count=high_risk,
        graduating_in_90_days=grad_90,
        at_risk_loan_value_cr=round(at_risk_loan, 2),
        risk_distribution=risk_dist,
        country_breakdown=country_bd,
        sector_heatmap=sector_hm,
    )


@router.get("/student/{student_id}/history")
async def student_history(student_id: str):
    """Simulated monthly risk score history for a student."""
    df = _load_data()
    student = df[df["student_id"] == student_id]

    if student.empty:
        raise HTTPException(404, f"Student {student_id} not found")

    row = student.iloc[0]
    base_score = float(row["placement_score"])
    risk_tier = row["risk_tier"]

    # Simulate 6-month history with slight variation
    rng = np.random.default_rng(hash(student_id) % 2**31)
    history = []
    for month in range(6, 0, -1):
        drift = float(rng.normal(0, 3))
        score = max(0, min(100, base_score + drift * (month / 6)))
        history.append({
            "month": f"T-{month}",
            "placement_score": round(score, 2),
            "risk_tier": "HIGH" if score < 35 else ("MEDIUM" if score < 55 else "LOW"),
        })

    # Current
    history.append({
        "month": "Current",
        "placement_score": round(base_score, 2),
        "risk_tier": risk_tier,
    })

    return {
        "student_id": student_id,
        "university": row["university_name"],
        "course": row["course_type"],
        "country": row["destination_country"],
        "graduation_year": int(row["graduation_year"]),
        "history": history,
    }
