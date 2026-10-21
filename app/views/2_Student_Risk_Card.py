"""
PlacementAI - Student Risk Card Page
Look up a student and review their risk tier, drivers, outcome, salary and actions.
"""

import os, sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ui import (
    ACCENT, MUTED, RISK_COLOR, chart, explain, inr, kpi_strip, lakh, load_data,
    model_rows, note, page_header, record_header, section, shap_section, value_rows,
)

df = load_data()

page_header("Student lookup", "Review one borrower's placement risk and the factors behind it.")

col_search, _, col_download = st.columns([1, 1, 1], vertical_alignment="bottom")
with col_search:
    search_id = st.text_input(
        "Student ID",
        value="STU000001",
        help="Format STU followed by six digits, from STU000001 to STU100000.",
    ).strip().upper()

if not search_id:
    st.stop()

student = df[df["student_id"] == search_id]
if student.empty:
    st.error(f"No student found with ID {search_id}. Check the ID and try again.")
    st.stop()

row = student.iloc[0]
risk = row["risk_tier"]
score = float(row["placement_score"])

record_header(
    row["student_id"],
    row["university_name"],
    f"{row['course_type']}, {row['destination_country']}, graduating {row['graduation_year']}",
    risk,
    "Placement score",
    f"{score:.1f}<small>/ 100</small>",
    tier_label="Recorded tier",
)

kpi_strip([
    ("CGPA", f"{row['cgpa']:.2f}", f"Institute tier {row['institute_tier']}"),
    ("Internships", str(row["internship_count"]), f"{row['internship_duration_months']} months in total"),
    ("Loan amount", lakh(row["loan_amount_inr"]), f"{row['loan_tenure_months']} month tenure"),
    ("Monthly EMI", f"₹{inr(row['emi_amount_inr'])}", f"Visa: {row['visa_type']}"),
])

# ── Model view and actual outcome ────────────────────────────────────────────
result, contrib, _ = explain(row.to_dict())

left, right = st.columns([3, 2], gap="large")

with left:
    section("Model assessment", "What the model predicts from this student's profile today.")
    model_rows(result)
    if result["risk_tier"] != risk:
        note(
            "The model's prediction differs from the recorded tier. Placement outcomes include "
            "random factors the profile cannot capture, which caps achievable accuracy near 80%."
        )

with right:
    section("Recorded placement outcome")
    value_rows([
        (f"Placed within {m} months", "Yes" if row[f"placed_{m}mo"] else "No",
         "down" if row[f"placed_{m}mo"] else "up")
        for m in (3, 6, 12)
    ])
    if isinstance(row.get("delay_reason"), str):
        note(f"Likely cause of delay: {row['delay_reason']}.")

shap_section(contrib)

# ── PDF risk card ────────────────────────────────────────────────────────────
from app.report import build_risk_card

actions = [a.strip() for a in str(row.get("recommended_actions", "")).split(";") if a.strip()]
salary_rows = None
if pd.notna(row.get("actual_salary_local")):
    salary_rows = [
        ("Role", str(row.get("job_role", ""))),
        (f"Annual salary ({row['salary_currency']})", f"{row['salary_currency']} {row['actual_salary_local']:,.0f}"),
        ("INR equivalent (PPP)", f"₹{inr(row['actual_salary_inr_ppp'])}"),
    ]
pdf = build_risk_card(
    record_id=row["student_id"],
    title=row["university_name"],
    meta=f"{row['course_type']}, {row['destination_country']}, graduating {row['graduation_year']}",
    tier_label="Recorded tier",
    tier=risk,
    score_label="Placement score",
    score_value=f"{score:.1f} / 100",
    facts=[
        ("CGPA", f"{row['cgpa']:.2f}"),
        ("Institute tier", str(row["institute_tier"])),
        ("Internships", str(row["internship_count"])),
        ("Loan amount", lakh(row["loan_amount_inr"])),
        ("Monthly EMI", f"₹{inr(row['emi_amount_inr'])}"),
        ("Tenure", f"{row['loan_tenure_months']} months"),
        ("Visa route", str(row["visa_type"])),
    ],
    result=result,
    contrib=contrib,
    outcome=[(f"Placed within {m} months", "Yes" if row[f"placed_{m}mo"] else "No") for m in (3, 6, 12)],
    salary=salary_rows,
    actions=actions,
)
with col_download:
    st.download_button(
        "Download risk card (PDF)", pdf, file_name=f"risk_card_{search_id}.pdf",
        mime="application/pdf", icon=":material/download:", on_click="ignore",
    )

# ── Salary and actions ───────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    section("Salary")
    if pd.notna(row.get("actual_salary_local")):
        currency = row.get("salary_currency", "USD")
        value_rows([
            ("Role", row.get("job_role", ""), "muted"),
            (f"Annual salary ({currency})", f"{currency} {row['actual_salary_local']:,.0f}", ""),
            ("INR equivalent (PPP)", f"₹{inr(row['actual_salary_inr_ppp'])}", ""),
            ("Salary to annual EMI", f"{row['actual_salary_inr_ppp'] / (row['emi_amount_inr'] * 12):.1f}x", ""),
        ])
    else:
        note("Not placed yet, so there is no salary on record.")

with right:
    section("Recommended actions")
    for action in actions:
        st.checkbox(action, key=f"action_{search_id}_{action}")
    if risk == "HIGH" and row["loan_amount_inr"] > 4_000_000:
        if st.button("Flag for EMI holiday review", type="primary"):
            st.success(f"{search_id} flagged for EMI holiday review.")

# ── History ──────────────────────────────────────────────────────────────────
section("Placement score, last six months", "Simulated monthly history. Dashed lines mark the tier thresholds.")

rng = np.random.default_rng(hash(search_id) % 2**31)
months, scores = [], []
for m in range(6, 0, -1):
    months.append(f"T-{m}")
    scores.append(round(max(0, min(100, score + float(rng.normal(0, 3)) * (m / 6))), 1))
months.append("Now")
scores.append(round(score, 1))

fig = go.Figure(go.Scatter(
    x=months, y=scores, mode="lines+markers",
    line=dict(color=ACCENT, width=2), marker=dict(size=6, color=ACCENT),
    hovertemplate="%{x}: %{y:.1f}<extra></extra>",
))
for y, tier, label in ((55, "MEDIUM", "Medium below 55"), (35, "HIGH", "High below 35")):
    fig.add_hline(y=y, line_dash="dot", line_width=1, line_color=RISK_COLOR[tier],
                  annotation_text=label, annotation_position="top left",
                  annotation_font=dict(size=11, color=MUTED))
fig.update_layout(showlegend=False)
fig.update_yaxes(range=[0, 100])
chart(fig, height=280)

with st.expander("Full record"):
    record = row.to_frame("Value")
    record["Value"] = record["Value"].astype(str)
    st.dataframe(record, width="stretch", height=420)
