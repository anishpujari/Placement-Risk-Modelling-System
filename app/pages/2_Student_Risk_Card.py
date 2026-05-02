"""
PlacementAI - Student Risk Card Page
Search student, display full risk card with SHAP, salary forecast, actions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Student Risk Card - PlacementAI", page_icon="🎯", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "placement_10k_final.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


st.title("🎯 Student Risk Card")
st.markdown("---")

df = load_data()

# Search
col_s1, col_s2 = st.columns([1, 3])
with col_s1:
    search_id = st.text_input("Student ID", value="STU00001", placeholder="e.g. STU00001")

if search_id:
    student = df[df["student_id"] == search_id]
    if student.empty:
        st.error(f"Student {search_id} not found.")
    else:
        row = student.iloc[0]
        risk = row["risk_tier"]
        score = row["placement_score"]

        # Color mapping
        risk_color = {"LOW": "#00b894", "MEDIUM": "#fdcb6e", "HIGH": "#e74c3c"}.get(risk, "#ccc")

        # ── Header ──
        h1, h2, h3 = st.columns([2, 1, 1])
        with h1:
            st.markdown(f"### {row['student_id']} — {row['university_name']}")
            st.markdown(f"**{row['course_type']}** | {row['destination_country']} | Graduating {row['graduation_year']}")
        with h2:
            st.markdown(f"<div style='background:{risk_color};color:white;padding:12px 20px;border-radius:12px;text-align:center;font-size:1.3rem;font-weight:700;'>{risk} RISK</div>", unsafe_allow_html=True)
        with h3:
            st.metric("Placement Score", f"{score:.1f}/100")

        st.markdown("---")

        # ── Risk Details ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CGPA", f"{row['cgpa']:.2f}")
        c2.metric("Internships", row['internship_count'])
        c3.metric("Loan Amount", f"₹{row['loan_amount_inr']/100000:.1f}L")
        c4.metric("EMI", f"₹{row['emi_amount_inr']:,.0f}/mo")

        st.markdown("---")

        # ── Placement Probabilities ──
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Placement Probability Timeline")
            probs = {
                "3 Months": row["placed_3mo"] * 100,
                "6 Months": row["placed_6mo"] * 100,
                "12 Months": row["placed_12mo"] * 100,
            }
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                marker_color=["#e74c3c" if v < 50 else "#fdcb6e" if v < 80 else "#00b894" for v in probs.values()],
                text=[f"{v:.0f}%" for v in probs.values()],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis_range=[0, 120], height=300,
                yaxis_title="Probability (%)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("SHAP Risk Drivers")
            # Display SHAP drivers as horizontal bars
            drivers = [row.get("shap_driver_1", ""), row.get("shap_driver_2", ""), row.get("shap_driver_3", "")]
            drivers = [d for d in drivers if d]

            if drivers:
                for i, d in enumerate(drivers):
                    icon = "🔴" if "+" in d else "🟢"
                    st.markdown(f"{icon} **{d}**")
            else:
                st.info("No SHAP drivers available")

        st.markdown("---")

        # ── Salary Forecast + Actions ──
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Salary Forecast")
            if pd.notna(row.get("actual_salary_local")):
                sal = row["actual_salary_local"]
                currency = row.get("salary_currency", "USD")
                ppp = row.get("actual_salary_inr_ppp", 0)
                st.metric(f"Actual Salary ({currency})", f"{currency} {sal:,.0f}")
                st.metric("INR PPP Equivalent", f"₹{ppp:,.0f}")
            else:
                st.warning("Not yet placed — salary data unavailable")

        with col_d:
            st.subheader("Recommended Actions")
            actions = str(row.get("recommended_actions", "")).split(";")
            for a in actions:
                a = a.strip()
                if a:
                    st.checkbox(a, key=f"action_{search_id}_{a}")

            if risk == "HIGH" and row["loan_amount_inr"] > 4000000:
                if st.button("🚨 Flag for EMI Holiday", type="primary"):
                    st.success(f"Student {search_id} flagged for EMI holiday review.")

        st.markdown("---")

        # ── Risk History ──
        st.subheader("Monthly Risk Score History (Simulated)")
        rng = np.random.default_rng(hash(search_id) % 2**31)
        history_scores = []
        for m in range(6, 0, -1):
            drift = float(rng.normal(0, 3))
            s = max(0, min(100, score + drift * (m / 6)))
            history_scores.append({"Month": f"T-{m}", "Score": round(s, 1)})
        history_scores.append({"Month": "Current", "Score": round(score, 1)})
        hist_df = pd.DataFrame(history_scores)

        fig_hist = px.line(
            hist_df, x="Month", y="Score",
            markers=True, line_shape="spline",
        )
        fig_hist.add_hline(y=55, line_dash="dash", line_color="#fdcb6e", annotation_text="MEDIUM threshold")
        fig_hist.add_hline(y=35, line_dash="dash", line_color="#e74c3c", annotation_text="HIGH threshold")
        fig_hist.update_layout(
            height=300,
            yaxis_range=[0, 100],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Full Details ──
        with st.expander("View Full Student Record"):
            st.json(row.to_dict())
