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
from sklearn.model_selection import train_test_split

from model.ceiling import ceiling_report
from model.predict import apply_high_threshold

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "placement_100k_final.csv")
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


TARGET_HIGH_RECALL = 0.80


def fit_model(X, y, verbose=False):
    """SMOTE-balance the given rows, then fit the XGBoost classifier."""
    X_res, y_res = SMOTE(random_state=42, k_neighbors=3).fit_resample(X, y)
    if verbose:
        print(f"After SMOTE: {len(X_res)} (from {len(X)})")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",  # CPU-optimized
        verbosity=0,
    )
    model.fit(X_res, y_res, verbose=False)
    model.n_train_after_smote_ = len(X_res)
    return model


def tune_high_threshold(proba, y_true, high_idx):
    """Highest P(HIGH) threshold whose validation HIGH recall still meets the target."""
    best = None
    for t in np.round(np.arange(0.20, 0.61, 0.01), 2):
        pred = apply_high_threshold(proba, t, high_idx)
        recall = ((pred == high_idx) & (y_true == high_idx)).sum() / (y_true == high_idx).sum()
        if recall >= TARGET_HIGH_RECALL:
            best = t
    return float(best) if best is not None else 0.20


def evaluate(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "high_risk_recall": round(report["HIGH"]["recall"], 4),
        "high_risk_precision": round(report["HIGH"]["precision"], 4),
        "per_class": {n: {k: round(v, 4) for k, v in report[n].items()} for n in class_names},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train():
    print("=" * 60)
    print("PlacementAI - Model Training Pipeline")
    print("=" * 60)

    # ── Load data ──
    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded {len(df)} records with {len(df.columns)} columns")

    raw_trend = df["sector_hiring_trend"].copy()
    raw_target = df[TARGET_COL].copy()

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

    # ── Validation slice (from training years only) for threshold tuning ──
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    high_idx = list(target_le.classes_).index("HIGH")

    print("\nFitting on 80% of training years to tune the HIGH-risk threshold...")
    val_model = fit_model(X_fit, y_fit)
    high_threshold = tune_high_threshold(val_model.predict_proba(X_val), y_val.values, high_idx)
    print(f"HIGH-risk threshold: {high_threshold:.2f} (target validation recall {TARGET_HIGH_RECALL:.0%})")

    # ── Final model on all training years ──
    print("\nTraining final XGBoost classifier on all training years...")
    model = fit_model(X_train, y_train, verbose=True)
    print("Training complete.")

    # ── Evaluate on the held-out 2025 cohort ──
    y_pred_proba = model.predict_proba(X_test)
    y_pred_argmax = y_pred_proba.argmax(axis=1)
    y_pred = apply_high_threshold(y_pred_proba, high_threshold, high_idx)

    class_names = list(target_le.classes_)
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc_roc = 0.0

    tuned = evaluate(y_test, y_pred, class_names)
    argmax = evaluate(y_test, y_pred_argmax, class_names)
    # Ceiling needs the original (unencoded) trend and tier labels
    ceiling = ceiling_report(
        df.loc[test_mask].assign(sector_hiring_trend=raw_trend[test_mask], risk_tier=raw_target[test_mask])
    )

    print(f"\n{'='*52}")
    print(f"  {'':22}{'Argmax':>10}{'Tuned':>10}")
    for key, label in [("accuracy", "Accuracy"), ("f1_macro", "F1 (macro)"), ("high_risk_recall", "HIGH recall"),
                       ("high_risk_precision", "HIGH precision")]:
        print(f"  {label:22}{argmax[key]:>10.4f}{tuned[key]:>10.4f}")
    print(f"  {'AUC-ROC':22}{auc_roc:>10.4f}")
    print(f"  {'Accuracy ceiling':22}{ceiling['oracle_accuracy']:>10.4f}")
    print(f"{'='*52}")
    print("\nClassification report (tuned decision rule):")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ── Save metrics ──
    metrics = {
        **{k: tuned[k] for k in ["accuracy", "f1_macro", "f1_weighted", "high_risk_recall", "high_risk_precision"]},
        "auc_roc": round(auc_roc, 4),
        "high_threshold": round(high_threshold, 2),
        "per_class": tuned["per_class"],
        "confusion_matrix": tuned["confusion_matrix"],
        "class_order": class_names,
        "argmax": argmax,
        "ceiling": ceiling,
        "train_size": len(X_train),
        "train_size_after_smote": int(model.n_train_after_smote_),
        "test_size": len(X_test),
    }
    with open(os.path.join(EVAL_DIR, "metrics_report.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "decision_config.json"), "w") as f:
        json.dump({"high_threshold": round(high_threshold, 2)}, f, indent=2)

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
