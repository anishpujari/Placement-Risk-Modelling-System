"""
PlacementAI - Salary Forecast Model (Quantile Regression)
Trains p10/p90 GradientBoosting regressors on placed students only.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "placement_100k_final.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

PPP_RATES = {"USD": 74, "CAD": 58, "GBP": 105, "AUD": 52}

SALARY_FEATURES = [
    "institute_tier", "cgpa", "internship_count", "internship_quality_tier",
    "certifications_count", "city_tier", "region_job_density",
    "visa_risk_score", "macro_condition_index",
]


def train_salary_models():
    """Train quantile regression models for salary forecasting."""
    print("Training salary forecast models...")

    df = pd.read_csv(DATA_PATH)
    placed = df[df["actual_salary_local"].notna()].copy()
    print(f"Placed students for training: {len(placed)}")

    # Encode course and country as numeric
    from sklearn.preprocessing import LabelEncoder
    le_course = LabelEncoder()
    le_country = LabelEncoder()
    placed["course_encoded"] = le_course.fit_transform(placed["course_type"])
    placed["country_encoded"] = le_country.fit_transform(placed["destination_country"])

    feature_cols = SALARY_FEATURES + ["course_encoded", "country_encoded"]
    X = placed[feature_cols]
    y = placed["actual_salary_local"]

    # Train p10 and p90
    model_p10 = GradientBoostingRegressor(
        loss="quantile", alpha=0.10, n_estimators=200, max_depth=5, random_state=42
    )
    model_p90 = GradientBoostingRegressor(
        loss="quantile", alpha=0.90, n_estimators=200, max_depth=5, random_state=42
    )

    model_p10.fit(X, y)
    model_p90.fit(X, y)

    # Evaluate
    pred_p10 = model_p10.predict(X)
    pred_p90 = model_p90.predict(X)
    pred_mid = (pred_p10 + pred_p90) / 2

    mae = mean_absolute_error(y, pred_mid)
    mape = mean_absolute_percentage_error(y, pred_mid)
    print(f"MAE: {mae:.0f} | MAPE: {mape:.2%}")

    # Save
    joblib.dump(model_p10, os.path.join(ARTIFACTS_DIR, "salary_model_p10.pkl"))
    joblib.dump(model_p90, os.path.join(ARTIFACTS_DIR, "salary_model_p90.pkl"))
    joblib.dump({"course": le_course, "country": le_country}, os.path.join(ARTIFACTS_DIR, "salary_encoders.pkl"))
    print("Salary models saved.")

    return model_p10, model_p90


def predict_salary(student_dict, model_p10=None, model_p90=None, salary_encoders=None):
    """Predict salary range for a single student."""
    if model_p10 is None:
        model_p10 = joblib.load(os.path.join(ARTIFACTS_DIR, "salary_model_p10.pkl"))
        model_p90 = joblib.load(os.path.join(ARTIFACTS_DIR, "salary_model_p90.pkl"))
        salary_encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "salary_encoders.pkl"))

    df = pd.DataFrame([student_dict])
    le_course = salary_encoders["course"]
    le_country = salary_encoders["country"]

    course_val = str(df["course_type"].iloc[0])
    country_val = str(df["destination_country"].iloc[0])
    df["course_encoded"] = le_course.transform([course_val])[0] if course_val in le_course.classes_ else 0
    df["country_encoded"] = le_country.transform([country_val])[0] if country_val in le_country.classes_ else 0

    feature_cols = SALARY_FEATURES + ["course_encoded", "country_encoded"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols]
    low = float(model_p10.predict(X)[0])
    high = float(model_p90.predict(X)[0])
    mid = (low + high) / 2

    currency_map = {"USA": "USD", "Canada": "CAD", "UK": "GBP", "Australia": "AUD"}
    currency = currency_map.get(country_val, "USD")
    ppp = PPP_RATES.get(currency, 74)

    return {
        "currency": currency,
        "range_low": int(round(low)),
        "range_high": int(round(high)),
        "inr_ppp_midpoint": int(round(mid * ppp)),
    }


if __name__ == "__main__":
    train_salary_models()
