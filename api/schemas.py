"""
PlacementAI - Pydantic v2 Request/Response Models
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ── Request Models ────────────────────────────────────────────────────────────

class StudentPredictRequest(BaseModel):
    student_id: str = Field(..., examples=["STU00001"])
    course_type: str = Field(..., examples=["Data Science/AI"])
    destination_country: str = Field(..., examples=["UK"])
    university_name: str = Field(..., examples=["University of Manchester"])
    institute_tier: int = Field(..., ge=1, le=4)
    cgpa: float = Field(..., ge=5.0, le=10.0)
    academic_consistency: str = Field(default="Steady", examples=["Steady"])
    internship_count: int = Field(default=0, ge=0, le=3)
    internship_duration_months: int = Field(default=0, ge=0)
    employer_type: str = Field(default="None", examples=["MNC"])
    internship_quality_tier: int = Field(default=0, ge=0, le=4)
    certifications_count: int = Field(default=0, ge=0, le=6)
    relevant_coursework_score: int = Field(default=3, ge=1, le=5)
    historic_placement_rate_3mo: float = Field(default=0.5, ge=0.0, le=1.0)
    historic_placement_rate_6mo: float = Field(default=0.6, ge=0.0, le=1.0)
    historic_placement_rate_12mo: float = Field(default=0.75, ge=0.0, le=1.0)
    placement_cell_score: int = Field(default=3, ge=1, le=5)
    recruiter_network_score: int = Field(default=3, ge=1, le=5)
    visa_type: str = Field(..., examples=["UK Graduate Route"])
    visa_risk_score: int = Field(default=2, ge=1, le=5)
    city_name: str = Field(default="Manchester", examples=["Manchester"])
    city_tier: int = Field(default=2, ge=1, le=3)
    region_job_density: int = Field(default=3, ge=1, le=5)
    target_sector: str = Field(default="Technology", examples=["Technology"])
    sector_hiring_trend: str = Field(default="Stable", examples=["Declining"])
    macro_condition_index: float = Field(default=3.5)
    job_portal_activity_score: int = Field(default=3, ge=1, le=5)
    interview_pipeline_stage: int = Field(default=0, ge=0, le=3)
    resume_updated_flag: int = Field(default=1, ge=0, le=1)
    skillup_event_flag: int = Field(default=0, ge=0, le=1)
    loan_amount_inr: int = Field(..., examples=[4800000])
    loan_tenure_months: int = Field(default=120, examples=[120])
    emi_amount_inr: int = Field(default=0)
    graduation_year: int = Field(..., examples=[2025])
    stem_flag: int = Field(default=0, ge=0, le=1)


# ── Response Models ───────────────────────────────────────────────────────────

class PlacementProbabilities(BaseModel):
    placed_3mo: float
    placed_6mo: float
    placed_12mo: float
    delayed_gt_12mo: float


class SalaryForecast(BaseModel):
    currency: str
    range_low: int
    range_high: int
    inr_ppp_midpoint: int


class ShapDriver(BaseModel):
    feature: str
    impact: str


class StudentPredictResponse(BaseModel):
    student_id: str
    risk_tier: str
    risk_score: int
    placement_probabilities: PlacementProbabilities
    salary_forecast: Optional[SalaryForecast] = None
    shap_drivers: list[ShapDriver] = []
    ai_risk_summary: str = ""
    recommended_actions: list[str] = []
    next_rescore_days: int = 30
    emi_holiday_flag: bool = False


class FraudFlag(BaseModel):
    rule_id: str
    rule_name: str
    severity: str
    description: str


class FraudCheckResponse(BaseModel):
    student_id: str
    flags: list[FraudFlag]


class PortfolioSummary(BaseModel):
    total_active_accounts: int
    high_risk_count: int
    graduating_in_90_days: int
    at_risk_loan_value_cr: float
    risk_distribution: dict
    country_breakdown: dict
    sector_heatmap: dict
