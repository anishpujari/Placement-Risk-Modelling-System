"""
PlacementAI — Synthetic Dataset Generator
Generates 100,000 Indian student records with 57 logically correlated columns.
"""

import numpy as np
import pandas as pd
from faker import Faker
import warnings, os
warnings.filterwarnings("ignore")

fake = Faker("en_IN")
Faker.seed(42)
np.random.seed(42)

N = 100_000

# ── Reference Tables ──────────────────────────────────────────────────────────

COURSE_TYPES = ["Engineering/CS", "Data Science/AI", "MBA/Business", "Finance", "Nursing/Healthcare"]
COURSE_PROBS = [0.30, 0.20, 0.20, 0.15, 0.15]

GRAD_YEARS = [2023, 2024, 2025]
GRAD_PROBS = [0.25, 0.40, 0.35]

COUNTRIES = ["USA", "Canada", "UK", "Australia"]
COUNTRY_PROBS = [0.32, 0.28, 0.22, 0.18]

MACRO_INDEX = {"USA": 4.1, "Canada": 3.8, "UK": 3.2, "Australia": 3.9}
PPP_RATES = {"USA": 74, "Canada": 58, "UK": 105, "Australia": 52}

# University names by country × tier (realistic QS-ranked)
UNIVERSITIES = {
    "USA": {
        1: ["MIT", "Stanford University", "Harvard University", "Caltech", "Carnegie Mellon University"],
        2: ["University of Michigan", "Georgia Tech", "UC San Diego", "University of Texas Austin", "NYU"],
        3: ["Arizona State University", "University of Florida", "Drexel University", "Northeastern University"],
        4: ["University of Bridgeport", "Harrisburg University", "Trine University", "Northwest Missouri State"],
    },
    "Canada": {
        1: ["University of Toronto", "UBC", "McGill University", "University of Waterloo"],
        2: ["University of Alberta", "McMaster University", "University of Ottawa", "Queen's University"],
        3: ["Dalhousie University", "University of Manitoba", "Carleton University", "Concordia University"],
        4: ["Lakehead University", "University of Northern BC", "Cape Breton University", "University of Regina"],
    },
    "UK": {
        1: ["University of Oxford", "University of Cambridge", "Imperial College London", "UCL"],
        2: ["University of Manchester", "University of Edinburgh", "King's College London", "University of Bristol"],
        3: ["University of Leicester", "Coventry University", "University of Sussex", "University of Kent"],
        4: ["University of Bedfordshire", "University of Bolton", "London Met University", "University of Sunderland"],
    },
    "Australia": {
        1: ["University of Melbourne", "University of Sydney", "ANU", "UNSW Sydney"],
        2: ["Monash University", "University of Queensland", "University of Adelaide", "UWA"],
        3: ["University of Wollongong", "RMIT University", "Curtin University", "Macquarie University"],
        4: ["Victoria University", "Charles Darwin University", "Southern Cross University", "CQUniversity"],
    },
}

CITIES = {
    "USA": {1: ["New York", "San Francisco", "Boston", "Seattle"], 2: ["Austin", "Chicago", "Atlanta", "Denver"], 3: ["Dayton", "Tulsa", "Springfield", "Harrisburg"]},
    "Canada": {1: ["Toronto", "Vancouver", "Montreal"], 2: ["Ottawa", "Calgary", "Edmonton"], 3: ["Winnipeg", "Halifax", "Regina"]},
    "UK": {1: ["London", "Manchester", "Edinburgh"], 2: ["Bristol", "Birmingham", "Leeds"], 3: ["Leicester", "Coventry", "Sunderland"]},
    "Australia": {1: ["Sydney", "Melbourne"], 2: ["Brisbane", "Adelaide", "Perth"], 3: ["Wollongong", "Darwin", "Cairns"]},
}

SECTOR_MAP = {
    "Engineering/CS": "Technology", "Data Science/AI": "Technology",
    "MBA/Business": "Consulting/Management", "Finance": "Banking/Finance",
    "Nursing/Healthcare": "Healthcare",
}

# sector_hiring_trend by country × sector
SECTOR_TRENDS = {
    ("USA", "Technology"): "Declining", ("USA", "Consulting/Management"): "Stable",
    ("USA", "Banking/Finance"): "Stable", ("USA", "Healthcare"): "Growing",
    ("Canada", "Technology"): "Growing", ("Canada", "Consulting/Management"): "Stable",
    ("Canada", "Banking/Finance"): "Stable", ("Canada", "Healthcare"): "Growing",
    ("UK", "Technology"): "Declining", ("UK", "Consulting/Management"): "Stable",
    ("UK", "Banking/Finance"): "Stable", ("UK", "Healthcare"): "Growing",
    ("Australia", "Technology"): "Growing", ("Australia", "Consulting/Management"): "Growing",
    ("Australia", "Banking/Finance"): "Stable", ("Australia", "Healthcare"): "Growing",
}

# Visa mapping
def get_visa(country, course):
    if country == "USA":
        return ("STEM-OPT", 1) if course in ["Engineering/CS", "Data Science/AI"] else ("OPT", np.random.choice([3, 4]))
    elif country == "Canada":
        return ("PGWP", np.random.choice([1, 2]))
    elif country == "UK":
        return ("UK Graduate Route", np.random.choice([2, 3]))
    else:
        return ("Temporary Graduate 485", np.random.choice([1, 2]))

# Base salaries (USD equivalent)
BASE_SALARY = {
    ("USA", "Engineering/CS"): 105000, ("USA", "Data Science/AI"): 115000,
    ("USA", "MBA/Business"): 95000, ("USA", "Finance"): 90000, ("USA", "Nursing/Healthcare"): 75000,
    ("Canada", "Engineering/CS"): 75000, ("Canada", "Data Science/AI"): 80000,
    ("Canada", "MBA/Business"): 70000, ("Canada", "Finance"): 65000, ("Canada", "Nursing/Healthcare"): 60000,
    ("UK", "Engineering/CS"): 55000, ("UK", "Data Science/AI"): 58000,
    ("UK", "MBA/Business"): 50000, ("UK", "Finance"): 52000, ("UK", "Nursing/Healthcare"): 42000,
    ("Australia", "Engineering/CS"): 80000, ("Australia", "Data Science/AI"): 82000,
    ("Australia", "MBA/Business"): 72000, ("Australia", "Finance"): 68000, ("Australia", "Nursing/Healthcare"): 58000,
}

JOB_ROLES = {
    "Engineering/CS": ["Software Engineer", "DevOps Engineer", "Systems Engineer", "Full Stack Developer"],
    "Data Science/AI": ["Data Scientist", "ML Engineer", "Data Analyst", "AI Researcher"],
    "MBA/Business": ["Management Consultant", "Business Analyst", "Product Manager", "Strategy Associate"],
    "Finance": ["Financial Analyst", "Investment Banking Analyst", "Risk Analyst", "Equity Research Associate"],
    "Nursing/Healthcare": ["Registered Nurse", "Clinical Nurse", "Healthcare Administrator", "Nurse Practitioner"],
}

DELAY_REASONS = ["Low internship exposure", "Weak job market", "Visa limitation", "Poor interview skills"]

ACTION_MAP = {
    "Technology": ["AWS/Azure certification", "Build GitHub portfolio", "Mock technical interviews"],
    "Banking/Finance": ["CFA Level 1 prep", "Bloomberg terminal certification", "LinkedIn Finance group networking"],
    "Consulting/Management": ["Case interview coaching", "MBA career fair participation", "MBA alumni outreach"],
    "Healthcare": ["Local licensing exam prep", "Hospital volunteer program", "Nursing agency registration"],
}


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_dataset():
    rng = np.random.default_rng(42)
    records = []

    for i in range(N):
        sid = f"STU{i+1:06d}"
        course = rng.choice(COURSE_TYPES, p=COURSE_PROBS)
        grad_year = int(rng.choice(GRAD_YEARS, p=GRAD_PROBS))
        country = rng.choice(COUNTRIES, p=COUNTRY_PROBS)

        # Institute tier — weighted by country (tier1 rarer)
        tier_probs = [0.15, 0.30, 0.30, 0.25]
        inst_tier = int(rng.choice([1, 2, 3, 4], p=tier_probs))

        uni_list = UNIVERSITIES[country][inst_tier]
        uni_name = rng.choice(uni_list)

        # CGPA correlated with tier
        cgpa_mean = {1: 8.3, 2: 7.6, 3: 7.0, 4: 6.4}[inst_tier]
        cgpa = float(np.clip(rng.normal(cgpa_mean, 0.7), 5.0, 10.0))
        cgpa = round(cgpa, 2)

        # Academic consistency
        ac_probs = {1: [0.75, 0.15, 0.10], 2: [0.70, 0.15, 0.15], 3: [0.65, 0.15, 0.20], 4: [0.60, 0.15, 0.25]}
        acad_consistency = rng.choice(["Steady", "Improving", "Declining"], p=ac_probs[inst_tier])

        # Internships correlated with tier
        intern_max = {1: 3, 2: 2, 3: 2, 4: 1}[inst_tier]
        intern_probs_map = {
            1: [0.05, 0.20, 0.40, 0.35], 2: [0.15, 0.35, 0.35, 0.15],
            3: [0.30, 0.35, 0.25, 0.10], 4: [0.50, 0.30, 0.15, 0.05],
        }
        intern_count = int(rng.choice([0, 1, 2, 3], p=intern_probs_map[inst_tier]))
        intern_duration = int(intern_count * rng.integers(2, 7)) if intern_count > 0 else 0

        # Employer type
        if intern_count == 0:
            emp_type = "None"
        else:
            emp_probs_map = {1: [0.55, 0.25, 0.20], 2: [0.35, 0.30, 0.35], 3: [0.15, 0.35, 0.50], 4: [0.10, 0.25, 0.65]}
            emp_type = rng.choice(["MNC", "Startup", "SME"], p=emp_probs_map[inst_tier])

        # Internship quality tier
        base_qual = {"MNC": 3, "Startup": 2, "SME": 1, "None": 0}[emp_type]
        intern_qual = int(np.clip(base_qual + rng.integers(-1, 2), 0, 4))

        # Certifications
        cert_mean = 3 if course in ["Data Science/AI", "Engineering/CS"] else 1.5
        certs = int(np.clip(rng.poisson(cert_mean), 0, 6))

        relevant_cw = int(np.clip(rng.integers(1, 6), 1, 5))

        # University features
        placement_3mo_base = {1: 0.75, 2: 0.55, 3: 0.35, 4: 0.20}[inst_tier]
        hist_3mo = round(float(np.clip(rng.normal(placement_3mo_base, 0.05), 0.05, 0.95)), 2)
        hist_6mo = round(float(np.clip(hist_3mo + rng.uniform(0.08, 0.18), hist_3mo, 0.98)), 2)
        hist_12mo = round(float(np.clip(hist_6mo + rng.uniform(0.05, 0.12), hist_6mo, 0.99)), 2)

        placement_cell = int(np.clip(round(rng.normal({1: 4.5, 2: 3.5, 3: 2.5, 4: 1.5}[inst_tier], 0.5)), 1, 5))
        recruiter_net = int(np.clip(round(rng.normal({1: 4.2, 2: 3.3, 3: 2.3, 4: 1.5}[inst_tier], 0.6)), 1, 5))

        # Visa
        visa_type, visa_risk = get_visa(country, course)

        # City
        city_tier = int(rng.choice([1, 2, 3], p=[0.40, 0.35, 0.25]))
        city_list = CITIES[country].get(city_tier, CITIES[country][1])
        city_name = rng.choice(city_list)
        region_job_density = int(np.clip(round(rng.normal({1: 4.2, 2: 3.0, 3: 1.8}[city_tier], 0.5)), 1, 5))

        sector = SECTOR_MAP[course]
        trend = SECTOR_TRENDS.get((country, sector), "Stable")
        macro = MACRO_INDEX[country]

        # Behavioral
        portal_activity = int(np.clip(rng.integers(1, 6), 1, 5))
        pipeline_stage = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.30, 0.25, 0.20]))
        resume_flag = int(rng.choice([0, 1], p=[0.30, 0.70]))
        skillup_flag = int(rng.choice([0, 1], p=[0.45, 0.55]))

        # Loan
        loan_base = {1: 35, 2: 42, 3: 50, 4: 55}[inst_tier]  # in lakhs
        loan_amount = int(np.clip(rng.normal(loan_base, 8), 18, 80) * 100000)
        tenure = int(rng.choice([96, 120, 144, 180]))
        monthly_rate = 0.0085
        emi = round(loan_amount * monthly_rate * (1 + monthly_rate) ** tenure / ((1 + monthly_rate) ** tenure - 1))
        grace_end = f"{grad_year}-{'06' if rng.random() > 0.5 else '12'}-30"

        # ── Placement Score ──
        inst_score = (5 - inst_tier) * 7
        intern_q_score = intern_qual * 6
        trend_score = {"Growing": 10, "Stable": 5, "Declining": 0}[trend]
        visa_inv = (6 - visa_risk) * 4
        high_w = inst_score + intern_q_score + trend_score + visa_inv

        cgpa_sc = (cgpa - 5) * 3
        cert_sc = certs * 2
        city_sc = (4 - city_tier) * 3
        med_w = cgpa_sc + cert_sc + city_sc

        portal_sc = portal_activity * 1.5
        pipe_sc = pipeline_stage * 5
        beh_w = portal_sc + pipe_sc

        raw_score = high_w + med_w + beh_w

        # Penalties
        if intern_count == 0 and inst_tier == 4 and cgpa < 6.5:
            raw_score *= 0.6
        if portal_activity >= 4 and pipeline_stage == 0:
            raw_score -= 5

        # Noise
        noise = float(np.clip(rng.normal(0, 8), -15, 15))
        placement_score = float(np.clip(raw_score + noise, 0, 100))

        # Fraud injection (~3%)
        is_fraud = False
        if rng.random() < 0.03:
            is_fraud = True
            placement_score = float(np.clip(placement_score + rng.choice([-20, 20]), 0, 100))

        placement_score = round(placement_score, 2)

        # Thresholds for placement labels
        t3, t6, t12 = 72, 55, 35
        placed_3mo = 1 if placement_score > t3 else 0
        placed_6mo = 1 if placement_score > t6 else 0
        placed_12mo = 1 if placement_score > t12 else 0

        # Monotonic constraint
        if placed_3mo:
            placed_6mo = 1
        if placed_6mo:
            placed_12mo = 1

        # Risk tier
        if placed_3mo == 1:
            risk_tier = "LOW"
        elif placed_6mo == 1:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "HIGH"

        # Salary (placed students only)
        if placed_12mo:
            base_sal = BASE_SALARY.get((country, course), 60000)
            tier_adj = {1: 1.20, 2: 1.05, 3: 0.95, 4: 0.85}[inst_tier]
            city_adj = {1: 1.10, 2: 1.0, 3: 0.90}[city_tier]
            adj_sal = base_sal * tier_adj * city_adj
            actual_sal = round(adj_sal * float(rng.uniform(0.85, 1.15)))
            currency = {"USA": "USD", "Canada": "CAD", "UK": "GBP", "Australia": "AUD"}[country]
            sal_inr_ppp = round(actual_sal * PPP_RATES[country])
            job_role = rng.choice(JOB_ROLES[course])
            final_emp = rng.choice(["MNC", "Startup", "SME"])
        else:
            actual_sal = None
            currency = None
            sal_inr_ppp = None
            job_role = None
            final_emp = None

        # Delay reason for HIGH risk
        delay_reason = rng.choice(DELAY_REASONS) if risk_tier == "HIGH" else None

        # SHAP drivers (simulated)
        drivers = []
        feat_impacts = [
            ("institute_tier", inst_score, "Institute tier", inst_tier),
            ("internship_quality_tier", intern_q_score, "Internship quality", intern_qual),
            ("sector_hiring_trend", trend_score, "Sector trend", trend),
            ("visa_risk_score", visa_inv, "Visa risk", visa_risk),
            ("cgpa", cgpa_sc, "CGPA", cgpa),
            ("job_portal_activity_score", portal_sc, "Job portal activity", portal_activity),
            ("interview_pipeline_stage", pipe_sc, "Interview pipeline", pipeline_stage),
            ("certifications_count", cert_sc, "Certifications", certs),
        ]
        sorted_feats = sorted(feat_impacts, key=lambda x: abs(x[1]), reverse=True)
        for feat_name, impact, label, val in sorted_feats[:3]:
            direction = "+" if (risk_tier != "LOW" and impact < 5) or (risk_tier == "LOW" and impact > 5) else "-"
            pct = abs(int(impact * 1.5))
            drivers.append(f"{label} ({val}) → {direction}{pct}% risk")

        shap1 = drivers[0] if len(drivers) > 0 else ""
        shap2 = drivers[1] if len(drivers) > 1 else ""
        shap3 = drivers[2] if len(drivers) > 2 else ""

        # Recommended actions
        actions = ACTION_MAP.get(sector, [])[:3]
        if risk_tier == "HIGH" and loan_amount > 4000000:
            actions.append("Flag for EMI holiday discussion")
        rec_actions = "; ".join(actions[:4])

        stem_flag = 1 if course in ["Engineering/CS", "Data Science/AI"] else 0

        records.append({
            "student_id": sid, "course_type": course, "graduation_year": grad_year,
            "destination_country": country, "university_name": uni_name, "institute_tier": inst_tier,
            "cgpa": cgpa, "academic_consistency": acad_consistency,
            "internship_count": intern_count, "internship_duration_months": intern_duration,
            "employer_type": emp_type, "internship_quality_tier": intern_qual,
            "certifications_count": certs, "relevant_coursework_score": relevant_cw,
            "historic_placement_rate_3mo": hist_3mo, "historic_placement_rate_6mo": hist_6mo,
            "historic_placement_rate_12mo": hist_12mo, "placement_cell_score": placement_cell,
            "recruiter_network_score": recruiter_net,
            "visa_type": visa_type, "visa_risk_score": visa_risk,
            "city_name": city_name, "city_tier": city_tier,
            "region_job_density": region_job_density,
            "target_sector": sector, "sector_hiring_trend": trend,
            "macro_condition_index": macro,
            "job_portal_activity_score": portal_activity, "interview_pipeline_stage": pipeline_stage,
            "resume_updated_flag": resume_flag, "skillup_event_flag": skillup_flag,
            "loan_amount_inr": loan_amount, "loan_tenure_months": tenure,
            "emi_amount_inr": emi, "grace_period_end_date": grace_end,
            "stem_flag": stem_flag,
            "placement_score": placement_score,
            "placed_3mo": placed_3mo, "placed_6mo": placed_6mo, "placed_12mo": placed_12mo,
            "risk_tier": risk_tier,
            "actual_salary_local": actual_sal, "salary_currency": currency,
            "actual_salary_inr_ppp": sal_inr_ppp,
            "job_role": job_role, "final_employer_type": final_emp,
            "delay_reason": delay_reason,
            "shap_driver_1": shap1, "shap_driver_2": shap2, "shap_driver_3": shap3,
            "recommended_actions": rec_actions,
            "is_fraud_flag": is_fraud,
        })

    df = pd.DataFrame(records)

    # ── Assertions ──
    assert len(df) == N, f"Expected {N} rows, got {len(df)}"
    assert df["placed_3mo"].le(df["placed_6mo"]).all(), "Monotonic violation: 3mo > 6mo"
    assert df["placed_6mo"].le(df["placed_12mo"]).all(), "Monotonic violation: 6mo > 12mo"
    assert df["cgpa"].between(5.0, 10.0).all(), "CGPA out of range"
    assert df["institute_tier"].between(1, 4).all(), "Tier out of range"

    dist = df["risk_tier"].value_counts(normalize=True)
    print("Risk distribution:")
    print(dist)
    print(f"\nTotal records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Fraud records: {df['is_fraud_flag'].sum()}")

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(out_dir, "placement_100k_final.csv"), index=False)
    df.sample(100, random_state=42).to_csv(os.path.join(out_dir, "placement_100k_sample100.csv"), index=False)
    print(f"\nDataset saved to {out_dir}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(f"\nSample columns ({len(df.columns)}):")
    print(list(df.columns))
