"""
PlacementAI - Baseline Model Comparison
Trains simpler models on the same features, split and SMOTE balancing as the production
XGBoost classifier, and compares them on the held-out 2025 cohort.

Run after model.train:  python -m model.compare_models
"""

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from model.ceiling import ceiling_report
from model.predict import apply_high_threshold, load_decision_threshold
from model.train import (
    ARTIFACTS_DIR, CATEGORICAL_COLS, DATA_PATH, EVAL_DIR,
    TARGET_COL, add_derived_features,
)

warnings.filterwarnings("ignore")

OUT_PATH = os.path.join(EVAL_DIR, "model_comparison.json")


def load_split():
    """Encode with the production encoders and split 2023-24 (train) vs 2025 (test)."""
    encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)

    raw = pd.read_csv(DATA_PATH)
    df = add_derived_features(raw)
    for col in CATEGORICAL_COLS:
        df[col] = encoders[col].transform(df[col].astype(str))
    y = encoders["__target__"].transform(df[TARGET_COL])

    train = (df["graduation_year"].isin([2023, 2024])).to_numpy()
    test = (df["graduation_year"] == 2025).to_numpy()
    X = df[feature_cols]
    classes = list(encoders["__target__"].classes_)
    return X[train], y[train], X[test], y[test], classes, raw[test]


def balance(X, y):
    """Same SMOTE settings as training; categorical codes are rounded back to valid integers."""
    X_res, y_res = SMOTE(random_state=42, k_neighbors=3).fit_resample(X, y)
    X_res[CATEGORICAL_COLS] = X_res[CATEGORICAL_COLS].round().astype(int)
    return X_res, y_res


def score(name, y_true, proba, classes, seconds, pred=None):
    high = classes.index("HIGH")
    pred = proba.argmax(axis=1) if pred is None else pred
    report = classification_report(y_true, pred, target_names=classes, output_dict=True, zero_division=0)
    try:
        auc = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
    except ValueError:
        auc = 0.5
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "f1_macro": round(f1_score(y_true, pred, average="macro", zero_division=0), 4),
        "auc_roc": round(auc, 4),
        "high_recall": round(report["HIGH"]["recall"], 4),
        "high_precision": round(report["HIGH"]["precision"], 4),
        "fit_seconds": round(seconds, 1),
        "_high_idx": high,
    }


def main():
    print("Loading data...")
    X_train, y_train, X_test, y_test, classes, raw_test = load_split()
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    X_bal, y_bal = balance(X_train, y_train)

    numeric = [c for c in X_train.columns if c not in CATEGORICAL_COLS]
    baselines = {
        "Logistic regression": make_pipeline(
            ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
                ("num", StandardScaler(), numeric),
            ]),
            LogisticRegression(max_iter=2000),
        ),
        "Decision tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=42),
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42
        ),
    }

    results = []
    majority = np.bincount(y_train).argmax()
    proba = np.zeros((len(y_test), len(classes)))
    proba[:, majority] = 1
    results.append(score(f"Always predict {classes[majority].title()}", y_test, proba, classes, 0.0))

    for name, model in baselines.items():
        print(f"Training {name}...")
        start = time.time()
        model.fit(X_bal, y_bal)
        results.append(score(name, y_test, model.predict_proba(X_test), classes, time.time() - start))

    print("Scoring production XGBoost...")
    xgb = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_classifier.pkl"))
    xgb_proba = xgb.predict_proba(X_test)
    results.append(score("XGBoost", y_test, xgb_proba, classes, float("nan")))
    threshold = load_decision_threshold()
    if threshold is not None:
        tuned = apply_high_threshold(xgb_proba, threshold, classes.index("HIGH"))
        results.append(score(f"XGBoost, tuned (P(high) >= {threshold:.2f})", y_test, xgb_proba, classes,
                             float("nan"), pred=tuned))

    for r in results:
        r.pop("_high_idx")
        if np.isnan(r["fit_seconds"]):
            r["fit_seconds"] = None

    ceiling = ceiling_report(raw_test)
    out = {
        "test_size": int(len(y_test)),
        "ceiling_accuracy": ceiling["oracle_accuracy"],
        "results": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    table = pd.DataFrame(results).set_index("model")
    print("\n" + table.to_string())
    print(f"\nAccuracy ceiling: {ceiling['oracle_accuracy']:.4f}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
