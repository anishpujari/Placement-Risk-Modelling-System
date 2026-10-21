"""
PlacementAI - Inference Module
Load trained model and generate predictions for new students.
"""

import os, json
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


DECISION_CONFIG = os.path.join(ARTIFACTS_DIR, "decision_config.json")


def apply_high_threshold(proba, threshold, high_idx):
    """
    Cost-sensitive decision rule: rate a student HIGH whenever P(HIGH) >= threshold,
    otherwise pick the more likely of the remaining tiers.
    """
    proba = np.atleast_2d(proba)
    others = proba.copy()
    others[:, high_idx] = -1
    pred = others.argmax(axis=1)
    pred[proba[:, high_idx] >= threshold] = high_idx
    return pred


def load_decision_threshold():
    """HIGH-risk threshold tuned during training; falls back to plain argmax if absent."""
    if os.path.exists(DECISION_CONFIG):
        with open(DECISION_CONFIG) as f:
            return json.load(f).get("high_threshold")
    return None


def load_model():
    """Load all model artifacts."""
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_classifier.pkl"))
    encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    return model, encoders, feature_cols


def prepare_features(student_dict, encoders, feature_cols):
    """Prepare a single student dict into model-ready features."""
    df = pd.DataFrame([student_dict])

    # Add derived features
    df["institute_internship_combined"] = (5 - df["institute_tier"]) * df.get("internship_quality_tier", 0)
    trend_map = {"Growing": 3, "Stable": 2, "Declining": 1}
    df["sector_trend_encoded"] = df["sector_hiring_trend"].map(trend_map).fillna(2)
    df["job_market_composite"] = df["sector_trend_encoded"] * df.get("region_job_density", 3) * df.get("macro_condition_index", 3.5)
    df["visa_risk_category_num"] = pd.cut(
        df["visa_risk_score"], bins=[0, 2, 3, 5], labels=[1, 2, 3]
    ).astype(float).fillna(2)
    df["loan_to_emi_ratio"] = df["loan_amount_inr"] / (df.get("emi_amount_inr", 30000) + 1)
    df["desperation_signal"] = ((df["job_portal_activity_score"] >= 4) & (df["interview_pipeline_stage"] == 0)).astype(int)

    # Encode categoricals
    cat_cols = ["course_type", "destination_country", "academic_consistency",
                "employer_type", "visa_type", "target_sector", "sector_hiring_trend"]
    for col in cat_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            val = str(df[col].iloc[0])
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = 0

    # Ensure all feature columns present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]


def predict_student(student_dict, model=None, encoders=None, feature_cols=None):
    """Generate full prediction for a single student."""
    if model is None:
        model, encoders, feature_cols = load_model()

    X = prepare_features(student_dict, encoders, feature_cols)
    proba = model.predict_proba(X)[0]
    target_le = encoders["__target__"]
    high_idx = list(target_le.classes_).index("HIGH")
    threshold = load_decision_threshold()
    if threshold is None:
        pred_class = int(np.argmax(proba))
    else:
        pred_class = int(apply_high_threshold(proba, threshold, high_idx)[0])

    risk_tier = target_le.inverse_transform([pred_class])[0]
    class_names = list(target_le.classes_)

    # Map probabilities to named tiers
    prob_map = {name: float(round(proba[i], 4)) for i, name in enumerate(class_names)}

    # Derive placement probabilities from risk tier probabilities
    p_low = prob_map.get("LOW", 0)
    p_med = prob_map.get("MEDIUM", 0)
    p_high = prob_map.get("HIGH", 0)

    placement_probs = {
        "placed_3mo": round(p_low, 4),
        "placed_6mo": round(p_low + p_med, 4),
        "placed_12mo": round(1.0 - p_high * 0.33, 4),
        "delayed_gt_12mo": round(p_high * 0.33, 4),
    }

    risk_score = int(round(p_high * 100 + p_med * 50))

    return {
        "risk_tier": risk_tier,
        "risk_score": risk_score,
        "placement_probabilities": placement_probs,
        "class_probabilities": prob_map,
    }
