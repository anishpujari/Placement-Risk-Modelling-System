"""
PlacementAI - Institute Benchmarking Page
University comparison table with placement rates, risk scores, and loan volumes.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Institute Benchmarking - PlacementAI", page_icon="🏫", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "placement_10k_final.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


st.title("🏫 Institute Benchmarking")
st.markdown("Compare university placement performance and identify underperforming institutes.")
st.markdown("---")

df = load_data()

# ── Filters ───────────────────────────────────────────────────────────────────
fc1, fc2, fc3 = st.columns(3)
with fc1:
    country_filter = st.multiselect("Country", df["destination_country"].unique().tolist(), default=df["destination_country"].unique().tolist())
with fc2:
    tier_filter = st.multiselect("Institute Tier", [1, 2, 3, 4], default=[1, 2, 3, 4])
with fc3:
    course_filter = st.multiselect("Course Type", df["course_type"].unique().tolist(), default=df["course_type"].unique().tolist())

filtered = df[
    df["destination_country"].isin(country_filter) &
    df["institute_tier"].isin(tier_filter) &
    df["course_type"].isin(course_filter)
]

# ── Aggregate by university ──────────────────────────────────────────────────
uni_stats = filtered.groupby("university_name").agg(
    country=("destination_country", "first"),
    tier=("institute_tier", "first"),
    students=("student_id", "count"),
    avg_placement_rate_6mo=("placed_6mo", "mean"),
    avg_placement_score=("placement_score", "mean"),
    avg_risk_score=("placement_score", lambda x: 100 - x.mean()),
    high_risk_pct=("risk_tier", lambda x: (x == "HIGH").mean() * 100),
    total_loan_cr=("loan_amount_inr", lambda x: x.sum() / 1e7),
    avg_cgpa=("cgpa", "mean"),
).reset_index()

uni_stats = uni_stats.round(2)
uni_stats = uni_stats.sort_values("avg_placement_rate_6mo", ascending=True)

# Flag underperforming
uni_stats["underperforming"] = uni_stats["avg_placement_rate_6mo"] < 0.40

st.markdown("---")

# ── KPIs ──
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Institutes", len(uni_stats))
k2.metric("Underperforming (<40%)", int(uni_stats["underperforming"].sum()))
k3.metric("Avg Placement Rate (6mo)", f"{uni_stats['avg_placement_rate_6mo'].mean()*100:.1f}%")
k4.metric("Total Loan Volume", f"₹{uni_stats['total_loan_cr'].sum():.0f} Cr")

st.markdown("---")

# ── Table ─────────────────────────────────────────────────────────────────────
st.subheader("University Performance Table")

def highlight_underperforming(row):
    if row.get("Underperforming", row.get("underperforming", False)):
        return ["background-color: rgba(231,76,60,0.15)"] * len(row)
    return [""] * len(row)

display_df = uni_stats.rename(columns={
    "university_name": "University",
    "country": "Country",
    "tier": "Tier",
    "students": "Students",
    "avg_placement_rate_6mo": "Placement Rate (6mo)",
    "avg_placement_score": "Avg Placement Score",
    "high_risk_pct": "HIGH Risk %",
    "total_loan_cr": "Loan Volume (Cr)",
    "avg_cgpa": "Avg CGPA",
    "underperforming": "Underperforming",
})

st.dataframe(
    display_df.style.apply(highlight_underperforming, axis=1),
    use_container_width=True,
    height=500,
)

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
ch1, ch2 = st.columns(2)

with ch1:
    st.subheader("Placement Rate by Institute Tier")
    tier_data = filtered.groupby("institute_tier")["placed_6mo"].mean().reset_index()
    tier_data.columns = ["Tier", "Placement Rate"]
    tier_data["Placement Rate"] *= 100
    fig = px.bar(
        tier_data, x="Tier", y="Placement Rate",
        color="Placement Rate",
        color_continuous_scale="RdYlGn",
        labels={"Placement Rate": "6-Month Placement Rate (%)"},
    )
    fig.update_layout(height=350, margin=dict(t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

with ch2:
    st.subheader("Risk Distribution by Tier")
    tier_risk = filtered.groupby(["institute_tier", "risk_tier"]).size().reset_index(name="count")
    colors = {"LOW": "#00b894", "MEDIUM": "#fdcb6e", "HIGH": "#e74c3c"}
    fig2 = px.bar(
        tier_risk, x="institute_tier", y="count", color="risk_tier",
        color_discrete_map=colors,
        barmode="stack",
        labels={"institute_tier": "Institute Tier", "count": "Students"},
    )
    fig2.update_layout(height=350, margin=dict(t=20, b=40))
    st.plotly_chart(fig2, use_container_width=True)
