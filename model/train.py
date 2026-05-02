"""
PlacementAI - Model Training Pipeline
XGBoost multi-class classifier with SMOTE, time-based split, and full evaluation.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, classification_report,
    roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "placement_10k_final.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ── Feature columns ──────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "course_type", "destination_country", "academic_consistency",
    "employer_type", "visa_type", "target_sector", "sector_hiring_trend",
]

NUMERICAL_COLS = [
    "institute_tier", "cgpa", "internship_count", "internship_duration_months",
    "internship_quality_tier", "certifications_count", "relevant_coursework_score",
    "historic_placement_rate_3mo", "historic_placement_rate_6mo",
    "historic_placement_rate_12mo", "placement_cell_score", "recruiter_network_score",
    "visa_risk_score", "city_tier", "region_job_density", "macro_condition_index",
    "job_portal_activity_score", "interview_pipeline_stage",
    "resume_updated_flag", "skillup_event_flag",
    "loan_amount_inr", "loan_tenure_months", "emi_amount_inr",
    "stem_flag",
]

TARGET_COL = "risk_tier"

# ── Derived features ─────────────────────────────────────────────────────────

def add_derived_features(df):
    """Add interaction and composite features."""
    df = df.copy()
    df["institute_internship_combined"] = (5 - df["institute_tier"]) * df["internship_quality_tier"]
    trend_map = {"Growing": 3, "Stable": 2, "Declining": 1}
    df["sector_trend_encoded"] = df["sector_hiring_trend"].map(trend_map).fillna(2)
    df["job_market_composite"] = df["sector_trend_encoded"] * df["region_job_density"] * df["macro_condition_index"]
    df["visa_risk_category_num"] = pd.cut(
        df["visa_risk_score"], bins=[0, 2, 3, 5], labels=[1, 2, 3]
    ).astype(float).fillna(2)
    df["loan_to_emi_ratio"] = df["loan_amount_inr"] / (df["emi_amount_inr"] + 1)
    df["desperation_signal"] = ((df["job_portal_activity_score"] >= 4) & (df["interview_pipeline_stage"] == 0)).astype(int)
    return df

DERIVED_COLS = [
    "institute_internship_combined", "sector_trend_encoded",
    "job_market_composite", "visa_risk_category_num",
    "loan_to_emi_ratio", "desperation_signal",
]


def train():
    print("=" * 60)
    print("PlacementAI - Model Training Pipeline")
    print("=" * 60)

    # ── Load data ──
    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded {len(df)} records with {len(df.columns)} columns")

    # ── Derived features ──
    df = add_derived_features(df)

    # ── Encode categoricals ──
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Target encoding
    target_le = LabelEncoder()
    df[TARGET_COL] = target_le.fit_transform(df[TARGET_COL])
    label_encoders["__target__"] = target_le
    print(f"Target classes: {list(target_le.classes_)}")

    # ── Feature set ──
    feature_cols = CATEGORICAL_COLS + NUMERICAL_COLS + DERIVED_COLS
    X = df[feature_cols]
    y = df[TARGET_COL]

    # ── Time-based split: 2023-2024 → train, 2025 → test ──
    train_mask = df["graduation_year"].isin([2023, 2024])
    test_mask = df["graduation_year"] == 2025
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # ── SMOTE on training set only ──
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_res)} (from {len(X_train)})")
    print(f"Class dist: {pd.Series(y_train_res).value_counts().to_dict()}")

    # ── Train XGBoost ──
    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",  # CPU-optimized
        verbosity=0,
    )
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print("Training complete.")

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # Per-class recall
    class_names = list(target_le.classes_)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # AUC-ROC (one-vs-rest)
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc_roc = 0.0

    # HIGH-risk recall
    high_idx = list(target_le.classes_).index("HIGH") if "HIGH" in target_le.classes_ else 0
    high_recall = report.get("HIGH", {}).get("recall", 0.0)

    print(f"\n{'='*40}")
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  F1 (macro):   {f1_macro:.4f}")
    print(f"  F1 (weighted):{f1_weighted:.4f}")
    print(f"  AUC-ROC:      {auc_roc:.4f}")
    print(f"  HIGH Recall:  {high_recall:.4f}")
    print(f"{'='*40}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ── Save metrics ──
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "auc_roc": round(auc_roc, 4),
        "high_risk_recall": round(high_recall, 4),
        "per_class": {name: {k: round(v, 4) for k, v in vals.items()} for name, vals in report.items() if name in class_names},
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_size": len(X_train_res),
        "test_size": len(X_test),
    }
    with open(os.path.join(EVAL_DIR, "metrics_report.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Save artifacts ──
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "xgb_classifier.pkl"))
    joblib.dump(label_encoders, os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")
    print(f"Metrics saved to {EVAL_DIR}")

    # ── Calibration plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 4))
        if len(class_names) == 1:
            axes = [axes]
        for idx, name in enumerate(class_names):
            y_binary = (y_test == idx).astype(int)
            prob_true, prob_pred = calibration_curve(y_binary, y_pred_proba[:, idx], n_bins=8)
            axes[idx].plot(prob_pred, prob_true, marker="o", label="Model")
            axes[idx].plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
            axes[idx].set_title(f"{name}")
            axes[idx].set_xlabel("Predicted Prob")
            axes[idx].set_ylabel("Actual Freq")
            axes[idx].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, "calibration_plot.png"), dpi=120)
        plt.close()
        print("Calibration plot saved.")
    except Exception as e:
        print(f"Calibration plot skipped: {e}")

    return model, label_encoders, feature_cols


if __name__ == "__main__":
    train()
