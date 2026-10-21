"""
PlacementAI - Accuracy ceiling for the synthetic dataset.

The generator labels each student from placement_score = deterministic score + random noise
(see data/generate_dataset.py). An oracle that knows the exact deterministic formula, but not
the noise, is the best any model can do. Its accuracy is the ceiling for this data.
"""

import numpy as np
import pandas as pd

# Must mirror the scoring block in data/generate_dataset.py
TREND_SCORE = {"Growing": 10, "Stable": 5, "Declining": 0}
THRESH_LOW, THRESH_MEDIUM = 72, 55


def deterministic_score(df):
    """Placement score before noise and fraud injection."""
    raw = (
        (5 - df["institute_tier"]) * 7
        + df["internship_quality_tier"] * 6
        + df["sector_hiring_trend"].map(TREND_SCORE)
        + (6 - df["visa_risk_score"]) * 4
        + (df["cgpa"] - 5) * 3
        + df["certifications_count"] * 2
        + (4 - df["city_tier"]) * 3
        + df["job_portal_activity_score"] * 1.5
        + df["interview_pipeline_stage"] * 5
    )
    penalty = (df["internship_count"] == 0) & (df["institute_tier"] == 4) & (df["cgpa"] < 6.5)
    raw = np.where(penalty, raw * 0.6, raw)
    raw = raw - np.where((df["job_portal_activity_score"] >= 4) & (df["interview_pipeline_stage"] == 0), 5, 0)
    return pd.Series(raw, index=df.index)


def oracle_tiers(df):
    raw = deterministic_score(df)
    return pd.Series(
        np.where(raw > THRESH_LOW, "LOW", np.where(raw > THRESH_MEDIUM, "MEDIUM", "HIGH")),
        index=df.index,
    )


def ceiling_report(df):
    """Oracle accuracy, per-class recall and noise level on the given rows."""
    oracle = oracle_tiers(df)
    actual = df["risk_tier"]
    return {
        "oracle_accuracy": round(float((oracle == actual).mean()), 4),
        "oracle_recall": {
            t: round(float((oracle[actual == t] == t).mean()), 4) for t in ["HIGH", "MEDIUM", "LOW"]
        },
        "noise_std": round(float((df["placement_score"] - deterministic_score(df)).std()), 2),
        "majority_baseline": round(float((actual == actual.value_counts().idxmax()).mean()), 4),
    }
