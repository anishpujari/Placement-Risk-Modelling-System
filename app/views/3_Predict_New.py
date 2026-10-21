"""
PlacementAI - Predict New Student Page
Assessment form for a new applicant, with live risk prediction, explanation and salary forecast.
"""

import os, sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ui import (
    driver_rows, explain, inr, kpi_strip, model_rows, note, page_header, record_header, section,
    shap_section, value_rows,
)

SECTOR_MAP = {
    "Engineering/CS": "Technology", "Data Science/AI": "Technology",
    "MBA/Business": "Consulting/Management", "Finance": "Banking/Finance",
    "Nursing/Healthcare": "Healthcare",
}
MACRO = {"USA": 4.1, "Canada": 3.8, "UK": 3.2, "Australia": 3.9}


page_header("New assessment", "Score an applicant before disbursal. All fields are required.")

with st.form("predict_form", border=False):
    section("Academic profile")
    a1, a2, a3, a4 = st.columns(4, gap="medium")
    with a1:
        student_id = st.text_input("Applicant reference", "NEW_001")
        course_type = st.selectbox("Course", list(SECTOR_MAP))
    with a2:
        university_name = st.text_input("University", "University of Toronto")
        institute_tier = st.selectbox("Institute tier", [1, 2, 3, 4], help="1 is the strongest tier.")
    with a3:
        cgpa = st.number_input("CGPA", 5.0, 10.0, 7.5, 0.1, format="%.1f")
        academic_consistency = st.selectbox("Academic trend", ["Steady", "Improving", "Declining"])
    with a4:
        graduation_year = st.number_input("Graduation year", 2023, 2026, 2025)
        certifications_count = st.number_input("Certifications", 0, 6, 2)

    section("Experience")
    e1, e2, e3, e4 = st.columns(4, gap="medium")
    with e1:
        internship_count = st.number_input("Internships", 0, 3, 1)
    with e2:
        internship_quality_tier = st.selectbox("Internship quality", [0, 1, 2, 3, 4], index=2,
                                               help="0 means no internship, 4 is the strongest.")
    with e3:
        employer_type = st.selectbox("Internship employer", ["None", "MNC", "Startup", "SME"], index=1)
    with e4:
        interview_pipeline = st.selectbox("Interview stage", [0, 1, 2, 3],
                                          help="0 means no interviews yet, 3 means final rounds.")

    section("Destination and loan")
    d1, d2, d3, d4 = st.columns(4, gap="medium")
    with d1:
        destination_country = st.selectbox("Destination country", list(MACRO))
        city_name = st.text_input("City", "Toronto")
    with d2:
        visa_type = st.selectbox("Visa route", ["STEM-OPT", "OPT", "PGWP", "UK Graduate Route", "Temporary Graduate 485"])
        city_tier = st.selectbox("City tier", [1, 2, 3])
    with d3:
        loan_amount = st.number_input("Loan amount (₹)", 1_800_000, 8_000_000, 4_500_000, 100_000)
        visa_risk_score = st.number_input("Visa risk score", 1, 5, 2, help="1 is the least restrictive.")
    with d4:
        loan_tenure = st.selectbox("Tenure (months)", [96, 120, 144, 180])
        sector_hiring_trend = st.selectbox("Sector hiring trend", ["Growing", "Stable", "Declining"])

    section("Job search activity")
    j1, j2, j3, j4 = st.columns(4, gap="medium")
    with j1:
        job_portal_activity = st.number_input("Job portal activity (1 to 5)", 1, 5, 3)
    with j2:
        region_job_density = st.number_input("Regional job density (1 to 5)", 1, 5, 3)
    with j3:
        resume_flag = st.selectbox("Resume updated recently", ["Yes", "No"])
    with j4:
        skillup_flag = st.selectbox("Recent skill-up activity", ["No", "Yes"])

    submitted = st.form_submit_button("Run assessment", type="primary")


if not submitted:
    st.stop()

from model.salary_model import predict_salary
from model.shap_explainer import describe_driver, generate_shap_summary

monthly_rate = 0.0085
emi = round(loan_amount * monthly_rate * (1 + monthly_rate) ** loan_tenure / ((1 + monthly_rate) ** loan_tenure - 1))

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
    "resume_updated_flag": 1 if resume_flag == "Yes" else 0,
    "skillup_event_flag": 1 if skillup_flag == "Yes" else 0,
    "loan_amount_inr": loan_amount,
    "loan_tenure_months": loan_tenure,
    "emi_amount_inr": emi,
    "stem_flag": 1 if course_type in ["Engineering/CS", "Data Science/AI"] else 0,
}

try:
    with st.spinner("Scoring applicant"):
        result, contrib, _ = explain(student_dict)
        drivers = [
            {"feature": describe_driver(r.feature, r.name, r.value, r.shap_high),
             "impact": f"{r.effect_pp:+.1f} pts"}
            for r in contrib.head(3).itertuples()
        ]
        try:
            salary = predict_salary(student_dict)
        except Exception:
            salary = None
except Exception as e:
    st.error(f"The assessment could not be completed: {e}")
    st.stop()

st.divider()

record_header(
    student_id,
    university_name,
    f"{course_type}, {destination_country}, graduating {graduation_year}",
    result["risk_tier"],
    "Risk score",
    f"{result['risk_score']}<small>/ 100</small>",
    tier_label="Predicted tier",
)

probs = result["placement_probabilities"]
kpi_strip([
    ("Placed within 3 months", f"{probs['placed_3mo'] * 100:.0f}%", None),
    ("Placed within 6 months", f"{probs['placed_6mo'] * 100:.0f}%", None),
    ("Placed within 12 months", f"{probs['placed_12mo'] * 100:.0f}%", None),
    ("Monthly EMI", f"₹{inr(emi)}", f"{loan_tenure} months at 10.2% a year"),
])

left, right = st.columns([3, 2], gap="large")

with left:
    section("Top drivers", "Largest SHAP contributions, as points of high-risk probability.")
    driver_rows([(d["feature"], d["impact"]) for d in drivers])
    section("Model assessment")
    model_rows(result)

with right:
    section("Expected salary")
    if salary:
        cur = salary["currency"]
        value_rows([
            (f"Low estimate ({cur})", f"{cur} {salary['range_low']:,}", ""),
            (f"High estimate ({cur})", f"{cur} {salary['range_high']:,}", ""),
            ("Midpoint in INR (PPP)", f"₹{inr(salary['inr_ppp_midpoint'])}", ""),
            ("Midpoint to annual EMI", f"{salary['inr_ppp_midpoint'] / (emi * 12):.1f}x", ""),
        ])
    else:
        note("The salary model is not available.")

shap_section(contrib)

from app.report import build_risk_card

pdf = build_risk_card(
    record_id=student_id,
    title=university_name,
    meta=f"{course_type}, {destination_country}, graduating {graduation_year}",
    tier_label="Predicted tier",
    tier=result["risk_tier"],
    score_label="Risk score",
    score_value=f"{result['risk_score']} / 100",
    facts=[
        ("CGPA", f"{cgpa:.1f}"),
        ("Institute tier", str(institute_tier)),
        ("Internships", str(internship_count)),
        ("Loan amount", f"₹{loan_amount / 1e5:.1f} L"),
        ("Monthly EMI", f"₹{inr(emi)}"),
        ("Tenure", f"{loan_tenure} months"),
        ("Visa route", visa_type),
    ],
    result=result,
    contrib=contrib,
    salary=[
        (f"Low estimate ({salary['currency']})", f"{salary['currency']} {salary['range_low']:,}"),
        (f"High estimate ({salary['currency']})", f"{salary['currency']} {salary['range_high']:,}"),
        ("Midpoint in INR (PPP)", f"₹{inr(salary['inr_ppp_midpoint'])}"),
    ] if salary else None,
)
st.download_button(
    "Download risk card (PDF)", pdf, file_name=f"risk_card_{student_id}.pdf",
    mime="application/pdf", icon=":material/download:", on_click="ignore",
)

with st.expander("Raw model output"):
    st.json({
        "risk_prediction": result,
        "shap_drivers": drivers,
        "ai_risk_summary": generate_shap_summary(drivers),
        "salary_forecast": salary,
    })
