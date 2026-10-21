"""
PlacementAI - Portfolio Overview Page
Headline figures, where high risk concentrates, exposure, and the priority account list.
"""

import os, sys

import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ui import (
    RISK_COLOR, RISK_LABEL, RISK_ORDER, chart, crore, inr, kpi_strip,
    load_data, page_header, section,
)

df = load_data()

total = len(df)
high = df["risk_tier"] == "HIGH"
high_count = int(high.sum())
grad_90 = int((df["graduation_year"] == 2025).sum() // 3)
exposure = df["loan_amount_inr"].sum()
high_exposure = df.loc[high, "loan_amount_inr"].sum()

page_header(
    "Portfolio overview",
    f"{inr(total)} active education loans across {df['destination_country'].nunique()} destination countries.",
)

kpi_strip([
    ("Active accounts", inr(total), f"Book size {crore(exposure)}"),
    ("High-risk accounts", inr(high_count), f"{high_count / total * 100:.1f}% of accounts"),
    ("Graduating in 90 days", inr(grad_90), "Estimated from the 2025 cohort"),
    ("Exposure at high risk", crore(high_exposure), f"{high_exposure / exposure * 100:.1f}% of book"),
])

risk_names = {k: RISK_LABEL[k] for k in RISK_ORDER}
risk_colors = {RISK_LABEL[k]: RISK_COLOR[k] for k in RISK_ORDER}

# ── Where high risk concentrates ─────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    section("High-risk share by sector and destination", "Percentage of accounts in each cell rated high risk.")
    pivot = (
        df.assign(is_high=high)
        .pivot_table(index="target_sector", columns="destination_country", values="is_high", aggfunc="mean")
        * 100
    ).round(1)
    fig = px.imshow(
        pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale=["#F7F1F0", "#E8B4AE", RISK_COLOR["HIGH"]],
    )
    fig.update_traces(hovertemplate="%{y}, %{x}: %{z:.1f}% high risk<extra></extra>", textfont=dict(size=13))
    fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(side="top", linecolor="rgba(0,0,0,0)")
    chart(fig, height=300)

with right:
    section("Risk mix by destination", "Share of accounts in each tier.")
    mix = (
        df.groupby("destination_country")["risk_tier"].value_counts(normalize=True)
        .mul(100).rename("share").reset_index()
    )
    order = mix[mix["risk_tier"] == "HIGH"].sort_values("share")["destination_country"].tolist()
    mix["Tier"] = mix["risk_tier"].map(risk_names)
    fig = px.bar(
        mix, y="destination_country", x="share", color="Tier", orientation="h",
        category_orders={"destination_country": order, "Tier": [RISK_LABEL[k] for k in RISK_ORDER]},
        color_discrete_map=risk_colors,
    )
    fig.update_traces(hovertemplate="%{y}: %{x:.1f}%<extra>%{fullData.name}</extra>")
    fig.update_layout(barmode="stack", xaxis_title=None, yaxis_title=None, bargap=0.45)
    fig.update_xaxes(ticksuffix="%", range=[0, 100], showgrid=True, gridcolor="#ECEFF2")
    chart(fig, height=300)

# ── Exposure and cohorts ─────────────────────────────────────────────────────
left, right = st.columns([2, 3], gap="large")

with left:
    section("Accounts by graduation cohort")
    cohort = df.groupby(["graduation_year", "risk_tier"]).size().rename("accounts").reset_index()
    cohort["Tier"] = cohort["risk_tier"].map(risk_names)
    fig = px.bar(
        cohort, x="graduation_year", y="accounts", color="Tier",
        category_orders={"Tier": [RISK_LABEL[k] for k in RISK_ORDER]},
        color_discrete_map=risk_colors,
    )
    fig.update_traces(hovertemplate="%{x}: %{y:,} accounts<extra>%{fullData.name}</extra>")
    fig.update_layout(barmode="stack", xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickformat=",.0f")
    chart(fig, height=300)

with right:
    section("Loan exposure by institute tier", "Outstanding principal in crore, split by risk tier.")
    exp = (
        df.groupby(["institute_tier", "risk_tier"])["loan_amount_inr"].sum().div(1e7)
        .rename("crore").reset_index()
    )
    exp["Tier"] = exp["risk_tier"].map(risk_names)
    exp["Institute"] = "Tier " + exp["institute_tier"].astype(str)
    fig = px.bar(
        exp, x="Institute", y="crore", color="Tier", barmode="group",
        category_orders={"Tier": [RISK_LABEL[k] for k in RISK_ORDER]},
        color_discrete_map=risk_colors,
    )
    fig.update_traces(hovertemplate="%{x}: ₹%{y:,.0f} Cr<extra>%{fullData.name}</extra>")
    fig.update_layout(xaxis_title=None, yaxis_title=None, bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(tickprefix="₹", ticksuffix=" Cr", tickformat=",.0f")
    chart(fig, height=300)

# ── Priority list ────────────────────────────────────────────────────────────
section(
    "Accounts to review first",
    "The 20 high-risk accounts with the lowest placement scores.",
)

priority = df[high].nsmallest(20, "placement_score")[[
    "student_id", "university_name", "course_type", "destination_country",
    "graduation_year", "placement_score", "loan_amount_inr", "delay_reason",
]].copy()
priority["loan_amount_inr"] = priority["loan_amount_inr"] / 1e5

st.dataframe(
    priority,
    hide_index=True,
    width="stretch",
    height=738,
    column_config={
        "student_id": st.column_config.TextColumn("Student"),
        "university_name": st.column_config.TextColumn("University"),
        "course_type": st.column_config.TextColumn("Course"),
        "destination_country": st.column_config.TextColumn("Country", width="small"),
        "graduation_year": st.column_config.NumberColumn("Grad year", format="%d", width="small"),
        "placement_score": st.column_config.NumberColumn("Score", format="%.1f", width="small"),
        "loan_amount_inr": st.column_config.NumberColumn("Loan", format="₹%.1f L", width="small"),
        "delay_reason": st.column_config.TextColumn("Likely cause of delay"),
    },
)
