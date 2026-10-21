"""
PlacementAI - SHAP Explainability Module
Per-student and portfolio-level SHAP explanations for the HIGH-risk class.

Each feature's effect is reported in percentage points of HIGH-risk probability:
the model's HIGH probability with all contributions, minus the probability with
that one feature's SHAP contribution removed from every class margin.
"""

import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

FEATURE_NAMES = {
    "course_type": "Course",
    "destination_country": "Destination country",
    "academic_consistency": "Academic trend",
    "employer_type": "Internship employer",
    "visa_type": "Visa route",
    "target_sector": "Target sector",
    "sector_hiring_trend": "Sector hiring trend",
    "institute_tier": "Institute tier",
    "cgpa": "CGPA",
    "internship_count": "Internships",
    "internship_duration_months": "Internship months",
    "internship_quality_tier": "Internship quality",
    "certifications_count": "Certifications",
    "relevant_coursework_score": "Relevant coursework",
    "historic_placement_rate_3mo": "Institute 3-month placement rate",
    "historic_placement_rate_6mo": "Institute 6-month placement rate",
    "historic_placement_rate_12mo": "Institute 12-month placement rate",
    "placement_cell_score": "Placement cell rating",
    "recruiter_network_score": "Recruiter network",
    "visa_risk_score": "Visa risk score",
    "city_tier": "City tier",
    "region_job_density": "Regional job density",
    "macro_condition_index": "Macro conditions",
    "job_portal_activity_score": "Job portal activity",
    "interview_pipeline_stage": "Interview stage",
    "resume_updated_flag": "Resume updated",
    "skillup_event_flag": "Recent skill-up",
    "loan_amount_inr": "Loan amount",
    "loan_tenure_months": "Loan tenure",
    "emi_amount_inr": "EMI",
    "stem_flag": "STEM course",
    "institute_internship_combined": "Institute x internship strength",
    "sector_trend_encoded": "Sector trend score",
    "job_market_composite": "Job market strength",
    "visa_risk_category_num": "Visa risk band",
    "loan_to_emi_ratio": "Loan to EMI ratio",
    "desperation_signal": "Searching with no interviews",
}

# Plain-English reading of a driver: (when it raises risk, when it lowers risk)
SHAP_TEXT_MAP = {
    "institute_tier": ("lower-tier institute", "higher-tier institute"),
    "internship_quality_tier": ("weak or no internship", "strong internship"),
    "sector_hiring_trend": ("sector hiring is weak", "sector is hiring"),
    "sector_trend_encoded": ("sector hiring is weak", "sector is hiring"),
    "visa_risk_score": ("restrictive visa route", "favourable visa route"),
    "visa_risk_category_num": ("restrictive visa route", "favourable visa route"),
    "cgpa": ("low CGPA", "strong CGPA"),
    "job_portal_activity_score": ("little job-search activity", "active job search"),
    "interview_pipeline_stage": ("few or no interviews", "interviews under way"),
    "certifications_count": ("few certifications", "relevant certifications"),
    "internship_count": ("little internship experience", "internship experience"),
    "institute_internship_combined": ("weak institute and internship mix", "strong institute and internship mix"),
    "job_market_composite": ("weak job market", "strong job market"),
    "city_tier": ("smaller job market city", "major job market city"),
    "region_job_density": ("few local openings", "many local openings"),
    "macro_condition_index": ("tough economy", "favourable economy"),
    "desperation_signal": ("applying widely with no interviews", "search is converting to interviews"),
}


@lru_cache(maxsize=1)
def high_class_index():
    """Index of the HIGH class in the model's output (classes are label-encoded alphabetically)."""
    encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"))
    return list(encoders["__target__"].classes_).index("HIGH")


def get_explainer(model=None):
    """Create SHAP TreeExplainer."""
    if model is None:
        model = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_classifier.pkl"))
    return shap.TreeExplainer(model)


def _as_3d(shap_values):
    """Normalise SHAP output to (rows, features, classes) across shap versions."""
    if isinstance(shap_values, list):  # shap < 0.45: one (rows, features) array per class
        return np.stack(shap_values, axis=-1)
    return shap_values


def _softmax(m):
    e = np.exp(m - m.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def explain_full(X_single, explainer, feature_names, student_data=None):
    """
    Every feature's contribution for one student.
    Returns (DataFrame sorted by absolute SHAP value, high-risk probability, baseline probability).
    DataFrame columns: feature, name, value, shap_high (log-odds), effect_pp (percentage points).
    """
    sv = _as_3d(explainer.shap_values(X_single))[0]          # (features, classes)
    base = np.asarray(explainer.expected_value, dtype=float)  # (classes,)
    h = high_class_index()

    margins = base + sv.sum(axis=0)
    p_high = _softmax(margins)[h]
    p_without = _softmax(margins[None, :] - sv)[:, h]          # drop one feature at a time
    effect_pp = (p_high - p_without) * 100

    values = []
    for i, f in enumerate(feature_names):
        if student_data is not None and f in student_data:
            values.append(student_data[f])
        else:
            values.append(np.asarray(X_single)[0, i])

    df = pd.DataFrame({
        "feature": feature_names,
        "name": [FEATURE_NAMES.get(f, f.replace("_", " ").capitalize()) for f in feature_names],
        "value": values,
        "shap_high": sv[:, h],
        "effect_pp": effect_pp,
    })
    df = df.reindex(df["shap_high"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return df, float(p_high * 100), float(_softmax(base)[h] * 100)


def _format_value(v):
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "None"
    if isinstance(v, (float, np.floating)) and not float(v).is_integer():
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{int(v):,}"
    return str(v)


def describe_driver(feature, name, value, shap_high):
    """One readable line for a driver, e.g. 'Interview stage: 0 (few or no interviews)'."""
    text = f"{name}: {_format_value(value)}"
    if feature in SHAP_TEXT_MAP:
        raises, lowers = SHAP_TEXT_MAP[feature]
        text += f" ({raises if shap_high > 0 else lowers})"
    return text


def explain_student(X_single, explainer, feature_names, student_data=None, top_n=3):
    """
    Top drivers of HIGH-risk probability for a single student.
    Returns list of dicts with feature (readable text), impact ('+4.2 pts'), raw_shap.
    """
    df, _, _ = explain_full(X_single, explainer, feature_names, student_data)
    drivers = []
    for r in df.head(top_n).itertuples():
        drivers.append({
            "feature": describe_driver(r.feature, r.name, r.value, r.shap_high),
            "impact": f"{r.effect_pp:+.1f} pts high-risk probability",
            "raw_shap": float(r.shap_high),
        })
    return drivers


def global_importance(X, explainer, feature_names):
    """
    Portfolio-level SHAP for the HIGH class.
    Returns (importance DataFrame sorted by mean |SHAP|, raw (rows, features) HIGH-class SHAP matrix).
    """
    sv = _as_3d(explainer.shap_values(X))[:, :, high_class_index()]
    imp = pd.DataFrame({
        "feature": feature_names,
        "name": [FEATURE_NAMES.get(f, f) for f in feature_names],
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return imp, sv


def generate_shap_summary(drivers):
    """Format SHAP drivers into a plain-English risk summary."""
    parts = [f"{d['feature']} ({d['impact']})" for d in drivers]
    return ". ".join(parts) + "." if parts else ""
