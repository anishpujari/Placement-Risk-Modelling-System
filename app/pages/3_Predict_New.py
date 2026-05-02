"""
PlacementAI - Predict New Student Page
Input form for all features, live prediction with full risk card + SHAP output.
"""

import streamlit as st
import os, sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Predict New Student - PlacementAI", page_icon="🔮", layout="wide")

st.title("🔮 Predict New Student")
st.markdown("Enter student details below for a live risk prediction.")
st.markdown("---")

# ── Input Form ────────────────────────────────────────────────────────────────

with st.form("predict_form"):
    st.subheader("Student Profile")
    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        student_id = st.text_input("Student ID", "NEW_001")
        course_type = st.selectbox("Course Type", [
            "Engineering/CS", "Data Science/AI", "MBA/Business", "Finance", "Nursing/Healthcare"
        ])
        graduation_year = st.number_input("Graduation Year", 2023, 2026, 2025)
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.1)

    with fc2:
        destination_country = st.selectbox("Destination Country", ["USA", "Canada", "UK", "Australia"])
        university_name = st.text_input("University Name", "University of Toronto")
        institute_tier = st.selectbox("Institute Tier", [1, 2, 3, 4])
        academic_consistency = st.selectbox("Academic Consistency", ["Steady", "Improving", "Declining"])

    with fc3:
        internship_count = st.number_input("Internship Count", 0, 3, 1)
        internship_quality_tier = st.selectbox("Internship Quality Tier", [0, 1, 2, 3, 4])
        certifications_count = st.number_input("Certifications", 0, 6, 2)
        employer_type = st.selectbox("Employer Type", ["None", "MNC", "Startup", "SME"])

    with fc4:
        visa_type = st.selectbox("Visa Type", ["STEM-OPT", "OPT", "PGWP", "UK Graduate Route", "Temporary Graduate 485"])
        visa_risk_score = st.slider("Visa Risk Score", 1, 5, 2)
        loan_amount = st.number_input("Loan Amount (INR)", 1800000, 8000000, 4500000, 100000)
        loan_tenure = st.selectbox("Loan Tenure (months)", [96, 120, 144, 180])

    st.subheader("Market & Behavioral")
    mc1, mc2, mc3, mc4 = st.columns(4)

    with mc1:
        city_name = st.text_input("City", "Toronto")
        city_tier = st.selectbox("City Tier", [1, 2, 3])

    with mc2:
        sector_hiring_trend = st.selectbox("Sector Hiring Trend", ["Growing", "Stable", "Declining"])
        region_job_density = st.slider("Region Job Density", 1, 5, 3)

    with mc3:
        job_portal_activity = st.slider("Job Portal Activity", 1, 5, 3)
        interview_pipeline = st.selectbox("Interview Pipeline Stage", [0, 1, 2, 3])

    with mc4:
        resume_flag = st.selectbox("Resume Updated?", [0, 1], index=1)
        skillup_flag = st.selectbox("Skill-up Events?", [0, 1])

    submitted = st.form_submit_button("🚀 Generate Prediction", type="primary", use_container_width=True)


# ── Prediction Output ─────────────────────────────────────────────────────────

if submitted:
    with st.spinner("Running AI prediction..."):
        try:
            from model.predict import predict_student, load_model, prepare_features
            from model.shap_explainer import get_explainer, explain_student, generate_shap_summary
            from model.salary_model import predict_salary

            SECTOR_MAP = {
                "Engineering/CS": "Technology", "Data Science/AI": "Technology",
                "MBA/Business": "Consulting/Management", "Finance": "Banking/Finance",
                "Nursing/Healthcare": "Healthcare",
            }

            monthly_rate = 0.0085
            emi = round(loan_amount * monthly_rate * (1 + monthly_rate) ** loan_tenure / ((1 + monthly_rate) ** loan_tenure - 1))
            MACRO = {"USA": 4.1, "Canada": 3.8, "UK": 3.2, "Australia": 3.9}

            student_dict = {
                "student_id": student_id,
                "course_type": course_type,
                "graduation_year": graduation_year,
                "destination_country": destination_country,
                "university_name": university_name,
                "institute_tier": institute_tier,
                "cgpa": cgpa,
                "academic_consistency": academic_consistency,
                "internship_count": internship_count,
                "internship_duration_months": internship_count * 3,
                "employer_type": employer_type,
                "internship_quality_tier": internship_quality_tier,
                "certifications_count": certifications_count,
                "relevant_coursework_score": 3,
                "historic_placement_rate_3mo": 0.5,
                "historic_placement_rate_6mo": 0.65,
                "historic_placement_rate_12mo": 0.8,
                "placement_cell_score": 3,
                "recruiter_network_score": 3,
                "visa_type": visa_type,
                "visa_risk_score": visa_risk_score,
                "city_name": city_name,
                "city_tier": city_tier,
                "region_job_density": region_job_density,
                "target_sector": SECTOR_MAP.get(course_type, "Technology"),
                "sector_hiring_trend": sector_hiring_trend,
                "macro_condition_index": MACRO.get(destination_country, 3.5),
                "job_portal_activity_score": job_portal_activity,
                "interview_pipeline_stage": interview_pipeline,
                "resume_updated_flag": resume_flag,
                "skillup_event_flag": skillup_flag,
                "loan_amount_inr": loan_amount,
                "loan_tenure_months": loan_tenure,
                "emi_amount_inr": emi,
                "stem_flag": 1 if course_type in ["Engineering/CS", "Data Science/AI"] else 0,
            }

            model, encoders, feature_cols = load_model()
            result = predict_student(student_dict, model, encoders, feature_cols)

            # SHAP
            X = prepare_features(student_dict, encoders, feature_cols)
            explainer = get_explainer(model)
            try:
                drivers = explain_student(X, explainer, feature_cols, student_dict)
            except Exception:
                drivers = [{"feature": "Explanation unavailable", "impact": "N/A"}]

            summary = generate_shap_summary(drivers)

            # Salary
            try:
                salary = predict_salary(student_dict)
            except Exception:
                salary = None

            # ── Display Results ──
            st.markdown("---")
            st.subheader("Prediction Results")

            risk = result["risk_tier"]
            risk_color = {"LOW": "#00b894", "MEDIUM": "#fdcb6e", "HIGH": "#e74c3c"}.get(risk, "#ccc")

            rc1, rc2, rc3 = st.columns([2, 1, 1])
            with rc1:
                st.markdown(f"### {student_id}")
                st.markdown(f"**{course_type}** | {destination_country} | {university_name}")
            with rc2:
                st.markdown(f"<div style='background:{risk_color};color:white;padding:12px 20px;border-radius:12px;text-align:center;font-size:1.3rem;font-weight:700;'>{risk} RISK</div>", unsafe_allow_html=True)
            with rc3:
                st.metric("Risk Score", result["risk_score"])

            st.markdown("---")

            # Probabilities
            pc1, pc2 = st.columns(2)
            with pc1:
                st.subheader("Placement Probabilities")
                probs = result["placement_probabilities"]
                for k, v in probs.items():
                    st.progress(min(v, 1.0), text=f"{k}: {v*100:.1f}%")

            with pc2:
                st.subheader("SHAP Drivers")
                for d in drivers:
                    icon = "🔴" if "+" in d.get("impact", "") else "🟢"
                    st.markdown(f"{icon} **{d['feature']}** ({d['impact']})")

            st.info(f"**AI Summary:** {summary}")

            # Salary
            if salary:
                st.subheader("Salary Forecast")
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric(f"Low ({salary['currency']})", f"{salary['range_low']:,}")
                sc2.metric(f"High ({salary['currency']})", f"{salary['range_high']:,}")
                sc3.metric("INR PPP Midpoint", f"₹{salary['inr_ppp_midpoint']:,}")

            # Full JSON
            with st.expander("View Full API Response"):
                output = {
                    "risk_prediction": result,
                    "shap_drivers": drivers,
                    "salary_forecast": salary,
                    "ai_risk_summary": summary,
                }
                st.json(output)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            import traceback
            st.code(traceback.format_exc())
