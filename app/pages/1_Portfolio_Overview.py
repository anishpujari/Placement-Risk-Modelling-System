"""
PlacementAI - Portfolio Overview Page
KPI cards, risk distribution, country×sector heatmap, alert table.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Portfolio Overview - PlacementAI", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "placement_10k_final.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


st.title("📊 Portfolio Overview")
st.markdown("---")

df = load_data()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total = len(df)
high_risk = int((df["risk_tier"] == "HIGH").sum())
grad_90 = int((df["graduation_year"] == 2025).sum() // 3)
at_risk_cr = round(df[df["risk_tier"] == "HIGH"]["loan_amount_inr"].sum() / 1e7, 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Active Accounts", f"{total:,}")
c2.metric("HIGH Risk Students", f"{high_risk:,}", delta=f"{high_risk/total*100:.1f}%", delta_color="inverse")
c3.metric("Graduating in 90 Days", f"{grad_90:,}")
c4.metric("At-Risk Loan Value", f"₹{at_risk_cr} Cr")

st.markdown("---")

# ── Risk Distribution + Country Breakdown ─────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Distribution")
    risk_counts = df["risk_tier"].value_counts()
    colors = {"LOW": "#00b894", "MEDIUM": "#fdcb6e", "HIGH": "#e74c3c"}
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.5,
        marker_colors=[colors.get(r, "#ccc") for r in risk_counts.index],
        textinfo="label+percent",
        textfont=dict(size=14),
    )])
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Country × Risk Distribution")
    country_risk = df.groupby(["destination_country", "risk_tier"]).size().reset_index(name="count")
    fig2 = px.bar(
        country_risk, x="destination_country", y="count", color="risk_tier",
        color_discrete_map=colors,
        barmode="stack",
        labels={"destination_country": "Country", "count": "Students"},
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(t=20, b=40),
        legend_title="Risk Tier",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Sector Heatmap ────────────────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Sector × Country Risk Heatmap")
    heatmap_data = df.groupby(["target_sector", "destination_country"]).apply(
        lambda x: (x["risk_tier"] == "HIGH").mean() * 100
    ).reset_index(name="high_risk_pct")
    pivot = heatmap_data.pivot(index="target_sector", columns="destination_country", values="high_risk_pct").fillna(0)
    fig3 = px.imshow(
        pivot.values,
        labels=dict(x="Country", y="Sector", color="HIGH Risk %"),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig3.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Graduating Cohort Timeline")
    cohort = df.groupby(["graduation_year", "risk_tier"]).size().reset_index(name="count")
    fig4 = px.bar(
        cohort, x="graduation_year", y="count", color="risk_tier",
        color_discrete_map=colors,
        barmode="group",
        labels={"graduation_year": "Graduation Year", "count": "Students"},
    )
    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── Alert Table: Top 20 Highest Risk ──────────────────────────────────────────
st.subheader("⚠️ Priority Alert — Top 20 Highest-Risk Students")

high_df = df[df["risk_tier"] == "HIGH"].nsmallest(20, "placement_score")
display_cols = [
    "student_id", "university_name", "course_type", "destination_country",
    "graduation_year", "placement_score", "risk_tier", "loan_amount_inr", "delay_reason",
]
st.dataframe(
    high_df[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=500,
)
