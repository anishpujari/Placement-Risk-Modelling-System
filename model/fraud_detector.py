"""
PlacementAI - Fraud Detection Module
Rule-based fraud flagging for anomalous institute-program-loan combos.
"""


FRAUD_RULES = [
    {
        "id": "FRAUD_001",
        "name": "Oversized loan for low-tier institute",
        "check": lambda r: r.get("loan_amount_inr", 0) > 7000000 and r.get("institute_tier", 1) >= 3,
        "severity": "HIGH",
        "description": "Loan amount exceeds Rs 70L for tier 3/4 institute - potential agent overcharging",
    },
    {
        "id": "FRAUD_002",
        "name": "Impossible visa-course combination",
        "check": lambda r: r.get("course_type") == "Nursing/Healthcare" and r.get("visa_type") == "STEM-OPT",
        "severity": "CRITICAL",
        "description": "Nursing student with STEM-OPT visa - STEM-OPT is only for STEM fields",
    },
    {
        "id": "FRAUD_003",
        "name": "Stale graduation year",
        "check": lambda r: r.get("graduation_year", 2025) < 2022,
        "severity": "MEDIUM",
        "description": "Graduation year before 2022 - potentially stale or recycled application",
    },
    {
        "id": "FRAUD_004",
        "name": "Improbable CGPA at low-tier institute",
        "check": lambda r: r.get("cgpa", 0) > 9.5 and r.get("institute_tier", 1) == 4,
        "severity": "MEDIUM",
        "description": "CGPA above 9.5 at tier 4 institute - statistically improbable, verify transcript",
    },
]


def check_fraud(student_dict):
    """
    Run all fraud rules against a student record.
    Returns list of triggered flags.
    """
    flags = []
    for rule in FRAUD_RULES:
        try:
            if rule["check"](student_dict):
                flags.append({
                    "rule_id": rule["id"],
                    "rule_name": rule["name"],
                    "severity": rule["severity"],
                    "description": rule["description"],
                })
        except Exception:
            continue
    return flags


def check_fraud_batch(records):
    """Run fraud checks on a list of student dicts. Returns flagged records."""
    results = []
    for rec in records:
        flags = check_fraud(rec)
        if flags:
            results.append({
                "student_id": rec.get("student_id", "UNKNOWN"),
                "flags": flags,
            })
    return results
