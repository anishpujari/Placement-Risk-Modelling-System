"""
PlacementAI - Institute Benchmarking Page
Compare universities on placement rate, risk and loan volume, and flag weak performers.
"""

import os, sys

import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ui import (
    ACCENT, RISK_COLOR, RISK_LABEL, RISK_ORDER, chart, crore, inr, kpi_strip, load_data,
    note, page_header, section,
)

UNDERPERFORMING = 0.40

df = load_data()

page_header("Institute benchmarking", "Compare universities and find the ones placing too few graduates.")

ALL = "All"
f1, f2, f3, _ = st.columns([1, 1, 1, 1], gap="medium")
with f1:
    country = st.selectbox("Destination country", [ALL] + sorted(df["destination_country"].unique()))
with f2:
    tier = st.selectbox("Institute tier", [ALL, 1, 2, 3, 4])
with f3:
    course = st.selectbox("Course", [ALL] + sorted(df["course_type"].unique()))

filtered = df
if country != ALL:
    filtered = filtered[filtered["destination_country"] == country]
if tier != ALL:
    filtered = filtered[filtered["institute_tier"] == tier]
if course != ALL:
    filtered = filtered[filtered["course_type"] == course]

if filtered.empty:
    note("No students match these filters. Widen one of them to see results.")
    st.stop()

uni = filtered.groupby("university_name").agg(
    country=("destination_country", "first"),
    tier=("institute_tier", "first"),
    students=("student_id", "count"),
    placed_6mo=("placed_6mo", "mean"),
    score=("placement_score", "mean"),
    high_pct=("risk_tier", lambda x: (x == "HIGH").mean() * 100),
    loan_cr=("loan_amount_inr", lambda x: x.sum() / 1e7),
    cgpa=("cgpa", "mean"),
).reset_index().sort_values("placed_6mo")
uni["placed_6mo"] *= 100
uni["status"] = uni["placed_6mo"].map(lambda r: "Below 40%" if r < UNDERPERFORMING * 100 else "")

weak = uni["status"] != ""
kpi_strip([
    ("Institutes", inr(len(uni)), f"{inr(len(filtered))} students"),
    ("Below 40% placement", inr(int(weak.sum())), f"{crore(uni.loc[weak, 'loan_cr'].sum() * 1e7)} lent to them"),
    ("Placed within 6 months", f"{filtered['placed_6mo'].mean() * 100:.1f}%", "Across selected students"),
    ("Loan volume", crore(filtered["loan_amount_inr"].sum()), None),
])

section("Institutes by six-month placement rate", "Weakest first. Institutes below 40% are marked.")
st.dataframe(
    uni,
    hide_index=True,
    width="stretch",
    height=460,
    column_order=["university_name", "status", "country", "tier", "students", "placed_6mo",
                  "high_pct", "score", "cgpa", "loan_cr"],
    column_config={
        "university_name": st.column_config.TextColumn("University"),
        "status": st.column_config.TextColumn("Flag", width="small"),
        "country": st.column_config.TextColumn("Country", width="small"),
        "tier": st.column_config.NumberColumn("Tier", format="%d", width="small"),
        "students": st.column_config.NumberColumn("Students", format="localized", width="small"),
        "placed_6mo": st.column_config.NumberColumn("Placed in 6 mo", format="%.1f%%"),
        "high_pct": st.column_config.NumberColumn("High risk", format="%.1f%%"),
        "score": st.column_config.NumberColumn("Avg score", format="%.1f", width="small"),
        "cgpa": st.column_config.NumberColumn("Avg CGPA", format="%.2f", width="small"),
        "loan_cr": st.column_config.NumberColumn("Loans", format="₹%.1f Cr"),
    },
)

left, right = st.columns(2, gap="large")

with left:
    section("Six-month placement rate by tier")
    by_tier = filtered.groupby("institute_tier")["placed_6mo"].mean().mul(100).reset_index()
    by_tier["Tier"] = "Tier " + by_tier["institute_tier"].astype(str)
    fig = px.bar(by_tier, x="Tier", y="placed_6mo", text="placed_6mo")
    fig.update_traces(marker_color=ACCENT, texttemplate="%{y:.1f}%", textposition="outside",
                      cliponaxis=False, hovertemplate="%{x}: %{y:.1f}%<extra></extra>")
    fig.add_hline(y=UNDERPERFORMING * 100, line_dash="dot", line_width=1, line_color=RISK_COLOR["HIGH"],
                  annotation_text="40% threshold", annotation_position="top right",
                  annotation_font=dict(size=11, color=RISK_COLOR["HIGH"]))
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    fig.update_yaxes(range=[0, 105], ticksuffix="%")
    chart(fig, height=320)

with right:
    section("Risk mix by tier")
    mix = filtered.groupby("institute_tier")["risk_tier"].value_counts(normalize=True).mul(100).rename("share").reset_index()
    mix["Tier"] = "Tier " + mix["institute_tier"].astype(str)
    mix["Risk"] = mix["risk_tier"].map(RISK_LABEL)
    fig = px.bar(
        mix, x="Tier", y="share", color="Risk",
        category_orders={"Risk": [RISK_LABEL[k] for k in RISK_ORDER]},
        color_discrete_map={RISK_LABEL[k]: RISK_COLOR[k] for k in RISK_ORDER},
    )
    fig.update_traces(hovertemplate="%{x}: %{y:.1f}%<extra>%{fullData.name}</extra>")
    fig.update_layout(barmode="stack", xaxis_title=None, yaxis_title=None)
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    chart(fig, height=320)
