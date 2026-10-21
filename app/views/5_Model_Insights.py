"""
PlacementAI - Model Insights Page
Held-out performance against the data's accuracy ceiling, the HIGH-risk threshold trade-off,
calibration, and portfolio-level SHAP explanations.
"""

import json
import os, sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ui import (
    ACCENT, BORDER, GRID, MUTED, RISK_COLOR, RISK_LABEL, RISK_ORDER, chart, kpi_strip, load_data,
    load_models, note, page_header, section,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_PATH = os.path.join(ROOT, "model", "evaluation", "metrics_report.json")
COMPARISON_PATH = os.path.join(ROOT, "model", "evaluation", "model_comparison.json")
SHAP_SAMPLE = 3000


@st.cache_data(show_spinner="Scoring the 2025 test cohort")
def test_predictions():
    from model.train import CATEGORICAL_COLS, add_derived_features
    model, encoders, feature_cols, _ = load_models()
    df = load_data()
    test = df[df["graduation_year"] == 2025].copy()
    raw = test.copy()
    test = add_derived_features(test)
    for col in CATEGORICAL_COLS:
        test[col] = encoders[col].transform(test[col].astype(str))
    X = test[feature_cols]
    proba = model.predict_proba(X)
    classes = list(encoders["__target__"].classes_)
    return X, proba, classes, raw


@st.cache_data(show_spinner="Computing SHAP values")
def global_shap():
    from model.shap_explainer import global_importance
    _, _, feature_cols, explainer = load_models()
    X, _, _, raw = test_predictions()
    sample = X.sample(min(SHAP_SAMPLE, len(X)), random_state=42)
    imp, sv = global_importance(sample, explainer, feature_cols)
    return imp, sv, sample, raw.loc[sample.index]


with open(METRICS_PATH) as f:
    m = json.load(f)
ceiling = m["ceiling"]

page_header(
    "Model insights",
    "How the risk model performs on the unseen 2025 cohort, and what it relies on.",
)

kpi_strip([
    ("Accuracy", f"{m['accuracy'] * 100:.1f}%", f"Ceiling for this data {ceiling['oracle_accuracy'] * 100:.1f}%"),
    ("High-risk recall", f"{m['high_risk_recall'] * 100:.1f}%", f"Target 79%, precision {m['high_risk_precision'] * 100:.1f}%"),
    ("AUC-ROC", f"{m['auc_roc']:.3f}", "Target 0.91"),
    ("F1 (macro)", f"{m['f1_macro'] * 100:.1f}%", f"Tested on {m['test_size']:,} students"),
])

# ── Accuracy ceiling and confusion matrix ───────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    section("Accuracy against the ceiling")
    bars = pd.DataFrame({
        "label": ["Always predict Low", "Model, tuned for high risk", "Model, most likely tier",
                  "Ceiling (exact formula, no noise)"],
        "acc": [ceiling["majority_baseline"], m["accuracy"], m["argmax"]["accuracy"], ceiling["oracle_accuracy"]],
    })
    bars["acc"] *= 100
    fig = go.Figure(go.Bar(
        x=bars["acc"], y=bars["label"], orientation="h",
        marker_color=["#A9B4C0", ACCENT, ACCENT, MUTED],
        text=[f"{v:.1f}%" for v in bars["acc"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(bargap=0.45)
    chart(fig, height=260)
    note(
        f"Each risk tier comes from a placement score plus random noise (standard deviation "
        f"{ceiling['noise_std']:.1f} points). Even a model that knew the exact scoring formula would "
        f"reach only {ceiling['oracle_accuracy'] * 100:.1f}%, so the model is within "
        f"{(ceiling['oracle_accuracy'] - m['argmax']['accuracy']) * 100:.1f} points of the best possible."
    )

with right:
    section("Confusion matrix", "Rows are the recorded tier, columns the model's tier. Cells show row share.")
    order = m["class_order"]
    cm = np.array(m["confusion_matrix"])
    idx = [order.index(t) for t in RISK_ORDER]
    cm = cm[np.ix_(idx, idx)]
    share = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = [RISK_LABEL[t] for t in RISK_ORDER]
    text = [[f"{share[i, j]:.0f}%" for j in range(3)] for i in range(3)]
    fig = go.Figure(go.Heatmap(
        z=share, x=labels, y=labels, text=text, texttemplate="%{text}", textfont=dict(size=14),
        customdata=cm, colorscale=[[0, "#F4F6F9"], [1, ACCENT]], showscale=False,
        hovertemplate="Recorded %{y}, predicted %{x}: %{z:.1f}% (%{customdata:,} students)<extra></extra>",
    ))
    fig.update_xaxes(side="top", title=dict(text="Predicted"), linecolor="rgba(0,0,0,0)")
    fig.update_yaxes(autorange="reversed", title=dict(text="Recorded"))
    chart(fig, height=300)

# ── Comparison with simpler models ───────────────────────────────────────────
section(
    "XGBoost against simpler models",
    "Same features, same 2023-24 training years, same SMOTE balancing, scored on the 2025 cohort.",
)
if os.path.exists(COMPARISON_PATH):
    with open(COMPARISON_PATH) as f:
        comp = json.load(f)
    table = pd.DataFrame(comp["results"])
    for col in ["accuracy", "f1_macro", "high_recall", "high_precision"]:
        table[col] *= 100
    table["model"] = table["model"].where(~table["model"].str.startswith("XGBoost, tuned"),
                                          "XGBoost, tuned for high risk")
    table["fit_seconds"] = table["fit_seconds"].map(
        lambda v: f"{v:.0f} s" if pd.notna(v) else "In main training")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_order=["model", "accuracy", "f1_macro", "auc_roc", "high_recall", "high_precision", "fit_seconds"],
        column_config={
            "model": st.column_config.TextColumn("Model", width="medium"),
            "accuracy": st.column_config.NumberColumn("Accuracy", format="%.1f%%"),
            "f1_macro": st.column_config.NumberColumn("F1 (macro)", format="%.1f%%"),
            "auc_roc": st.column_config.NumberColumn("AUC-ROC", format="%.3f"),
            "high_recall": st.column_config.NumberColumn("High-risk recall", format="%.1f%%"),
            "high_precision": st.column_config.NumberColumn("High-risk precision", format="%.1f%%"),
            "fit_seconds": st.column_config.TextColumn("Training time"),
        },
    )
    baselines = table[~table["model"].str.startswith(("XGBoost", "Always"))]
    runner_up = baselines.loc[baselines["accuracy"].idxmax()]
    note(
        f"XGBoost has the best accuracy, F1 and AUC. {runner_up['model']} is the closest at "
        f"{runner_up['accuracy']:.1f}% accuracy, which is expected because the placement score behind "
        f"the labels is largely a weighted sum of the features. Only the tuned XGBoost meets the 79% "
        f"high-risk recall target. All models sit below the {comp['ceiling_accuracy'] * 100:.1f}% ceiling."
    )
else:
    note("Run `python -m model.compare_models` to generate the comparison.")

# ── Threshold trade-off and calibration ──────────────────────────────────────
X_test, proba, classes, raw_test = test_predictions()
high_idx = classes.index("HIGH")
y_high = (raw_test["risk_tier"] == "HIGH").to_numpy()
p_high = proba[:, high_idx]

left, right = st.columns([1, 1], gap="large")

with left:
    section(
        "High-risk threshold trade-off",
        f"Flagging a student as high risk once P(high) passes {m['high_threshold']:.2f} catches more risky "
        "borrowers at the cost of some false alarms. The threshold was chosen on validation data.",
    )
    ts = np.round(np.arange(0.10, 0.81, 0.02), 2)
    rec = [((p_high >= t) & y_high).sum() / y_high.sum() * 100 for t in ts]
    prec = [((p_high >= t) & y_high).sum() / max((p_high >= t).sum(), 1) * 100 for t in ts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=rec, name="Recall", line=dict(color=RISK_COLOR["HIGH"], width=2),
                             hovertemplate="Threshold %{x:.2f}: recall %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=ts, y=prec, name="Precision", line=dict(color=ACCENT, width=2),
                             hovertemplate="Threshold %{x:.2f}: precision %{y:.1f}%<extra></extra>"))
    fig.add_vline(x=m["high_threshold"], line_dash="dot", line_width=1, line_color=MUTED,
                  annotation_text=f"Chosen {m['high_threshold']:.2f}", annotation_position="top right",
                  annotation_font=dict(size=11, color=MUTED))
    fig.update_xaxes(title=dict(text="P(high) threshold"), showgrid=True, gridcolor=GRID)
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    chart(fig, height=300)

with right:
    section("Calibration of high-risk probability",
            "When the model says 30%, about 30% of those students should be high risk.")
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(p_high, bins) - 1, 0, 9)
    cal = pd.DataFrame({"bin": which, "p": p_high, "y": y_high}).groupby("bin").agg(
        predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect",
                             line=dict(color=BORDER, dash="dot", width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cal["predicted"] * 100, y=cal["actual"] * 100, mode="lines+markers",
                             name="Model", line=dict(color=ACCENT, width=2), marker=dict(size=7),
                             customdata=cal["n"],
                             hovertemplate="Predicted %{x:.0f}%, actual %{y:.0f}% (%{customdata:,} students)<extra></extra>"))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title=dict(text="Predicted P(high)"),
                     showgrid=True, gridcolor=GRID)
    fig.update_yaxes(range=[0, 100], ticksuffix="%", title=dict(text="Actual high-risk share"))
    chart(fig, height=300)
    note("Training used SMOTE to balance the tiers, which inflates predicted high-risk probability "
         "relative to the real 17% share. Rankings and the tuned threshold are unaffected.")

# ── Global SHAP ──────────────────────────────────────────────────────────────
imp, sv, sample, raw_sample = global_shap()

left, right = st.columns([2, 3], gap="large")

with left:
    section("What the model relies on",
            f"Mean absolute SHAP value for the high-risk class over {len(sample):,} test students.")
    top = imp.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["mean_abs_shap"], y=top["name"], orientation="h", marker_color=ACCENT,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title=dict(text="Mean |SHAP| (log-odds)"), showgrid=True, gridcolor=GRID, nticks=4)
    fig.update_layout(bargap=0.3)
    chart(fig, height=520)

with right:
    section("How one feature moves risk",
            "Each point is a student. Points above zero push toward high risk.")
    names = imp.head(12)["name"].tolist()
    chosen = st.selectbox("Feature", names, index=0, label_visibility="collapsed")
    feat = imp.loc[imp["name"] == chosen, "feature"].iloc[0]
    j = list(sample.columns).index(feat)
    x_vals = raw_sample[feat] if feat in raw_sample.columns else sample[feat]
    y_vals = sv[:, j]
    plot_df = pd.DataFrame({"x": x_vals.astype(str) if x_vals.dtype == object else x_vals, "shap": y_vals})
    if plot_df["x"].dtype != object and plot_df["x"].nunique() <= 12:
        rng = np.random.default_rng(0)
        plot_df["x"] = plot_df["x"] + rng.uniform(-0.18, 0.18, len(plot_df))
    colors = np.where(plot_df["shap"] > 0, RISK_COLOR["HIGH"], RISK_COLOR["LOW"])
    fig = go.Figure(go.Scattergl(
        x=plot_df["x"], y=plot_df["shap"], mode="markers",
        marker=dict(size=5, color=colors, opacity=0.45),
        hovertemplate=f"{chosen}: %{{x}}<br>SHAP %{{y:+.3f}}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=BORDER)
    fig.update_xaxes(title=dict(text=chosen), showgrid=False)
    fig.update_yaxes(title=dict(text="SHAP for high risk"))
    chart(fig, height=470)
