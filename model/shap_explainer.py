"""
PlacementAI - SHAP Explainability Module
Generate per-student top-3 SHAP drivers with plain-English explanations.
"""

import numpy as np
import shap
import joblib, os, json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

SHAP_TEXT_MAP = {
    "institute_tier": ("Institute tier {val}", "higher-tier boosts placement", "lower-tier increases risk"),
    "internship_quality_tier": ("Internship quality tier {val}", "strong internship reduces risk", "weak/no internship increases risk"),
    "sector_hiring_trend": ("Sector trend: {val}", "growing sector helps", "declining sector adds risk"),
    "visa_risk_score": ("Visa type risk score {val}", "favorable visa reduces risk", "visa constraints increase risk"),
    "cgpa": ("CGPA {val}", "strong CGPA reduces risk", "low CGPA increases risk"),
    "job_portal_activity_score": ("Job portal activity {val}/5", "active job search helps", "low activity is concerning"),
    "interview_pipeline_stage": ("Interview stage: {val}", "active pipeline reduces risk", "no pipeline is a warning sign"),
    "certifications_count": ("Certifications: {val}", "certifications reduce risk", "few certifications noted"),
    "internship_count": ("Internships: {val}", "internship experience helps", "no internship increases risk"),
    "institute_internship_combined": ("Institute x Internship score", "strong combo reduces risk", "weak combo adds risk"),
    "job_market_composite": ("Job market composite", "strong market helps", "weak market adds risk"),
    "loan_amount_inr": ("Loan amount", "moderate loan is manageable", "high loan increases exposure"),
    "desperation_signal": ("Desperation signal", "", "high activity + no interviews is a warning"),
    "historic_placement_rate_3mo": ("3-month placement rate", "strong rate reduces risk", "weak rate increases risk"),
    "city_tier": ("City tier {val}", "major city helps placement", "smaller city limits opportunities"),
    "macro_condition_index": ("Macro conditions", "favorable conditions", "challenging macro environment"),
    "region_job_density": ("Regional job density", "strong job market", "limited job market"),
    "destination_country": ("Destination country", "favorable market", "challenging market"),
    "course_type": ("Course type", "in-demand course", "course with limited demand"),
}


def get_explainer(model=None):
    """Create SHAP TreeExplainer."""
    if model is None:
        model = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_classifier.pkl"))
    return shap.TreeExplainer(model)


def explain_student(X_single, explainer, feature_names, student_data=None):
    """
    Generate top-3 SHAP drivers for a single student.
    X_single: 1-row DataFrame or array with features
    Returns list of dicts with feature, impact, direction.
    """
    shap_values = explainer.shap_values(X_single)

    # For multiclass, shap_values is a list of arrays (one per class)
    # We focus on the predicted class or use max absolute impact across classes
    if isinstance(shap_values, list):
        # Sum absolute SHAP across classes for feature importance
        combined = np.sum([np.abs(sv) for sv in shap_values], axis=0)
        shap_row = combined[0] if combined.ndim > 1 else combined
        # Get direction from HIGH risk class (usually last or index of HIGH)
        high_class_shap = shap_values[-1][0] if shap_values[-1].ndim > 1 else shap_values[-1]
    else:
        shap_row = np.abs(shap_values[0]) if shap_values.ndim > 1 else np.abs(shap_values)
        high_class_shap = shap_values[0] if shap_values.ndim > 1 else shap_values

    # Get top features by absolute SHAP value
    top_indices = np.argsort(shap_row)[::-1][:3]

    drivers = []
    for idx in top_indices:
        feat_name = feature_names[idx]
        abs_impact = float(shap_row[idx])
        direction = "positive" if high_class_shap[idx] > 0 else "negative"
        pct = int(round(abs_impact * 15))  # scale to percentage

        # Get readable text
        if feat_name in SHAP_TEXT_MAP:
            template, pos_text, neg_text = SHAP_TEXT_MAP[feat_name]
            val = ""
            if student_data is not None and feat_name in student_data:
                val = student_data[feat_name]
            label = template.format(val=val)
            explanation = neg_text if direction == "positive" else pos_text
        else:
            label = feat_name.replace("_", " ").title()
            explanation = "increases risk" if direction == "positive" else "reduces risk"

        sign = "+" if direction == "positive" else "-"
        drivers.append({
            "feature": f"{label} - {explanation}",
            "impact": f"{sign}{pct}% risk",
            "raw_shap": float(high_class_shap[idx]),
        })

    return drivers


def generate_shap_summary(drivers):
    """Format SHAP drivers into a plain-English risk summary."""
    parts = []
    for d in drivers:
        parts.append(f"{d['feature']} ({d['impact']})")
    return ". ".join(parts) + "."
