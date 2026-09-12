"""
Phase 7 — Inference API (Section 18/19): Pydantic request/response
models for POST /api/v1/predict.

Design note on field naming: `PredictRequest`'s fields intentionally
reuse the EXACT raw dataset column names from `src/data/schema.py`
(Age, Gender, Academic_Level, ...) rather than idiomatic snake_case
API field names. This is a deliberate trade-off: a snake_case ->
PascalCase mapping layer is exactly the kind of hand-rolled
translation step that Section 18 warns causes training/serving skew,
so it's left out entirely. `request.model_dump()` can be passed
straight into `pd.DataFrame([...])` and from there into
`add_engineered_features` / `encoder.transform` unchanged.

Only the 9 raw fields the model actually consumes are exposed
(`Student_ID`, `Addicted_Score`, `Affects_Academic_Performance`,
`Mental_Health_Score` are ID/leakage/target columns per
`src/models/baseline.py::LEAKAGE_AND_ID_COLS` and are never inputs).

Range/category bounds mirror `src/data/schema.py`'s Pandera contract,
with one intentional tightening: `Sleep_Hours_Per_Night` is `gt=0`
here (the raw ingestion schema allows `ge=0`) because a literal 0
would divide-by-zero in `usage_to_sleep_ratio`
(`src/features/engineering.py`) — a value the training data never
actually contained, so rejecting it at the API boundary (422) is
correct per Section 18: "malformed input -> 422 before the model is
ever touched."
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.data.schema import (
    KNOWN_ACADEMIC_LEVELS,
    KNOWN_GENDERS,
    KNOWN_PLATFORMS,
    KNOWN_RELATIONSHIP_STATUSES,
)

Gender = Literal[tuple(KNOWN_GENDERS)]  # type: ignore[valid-type]
AcademicLevel = Literal[tuple(KNOWN_ACADEMIC_LEVELS)]  # type: ignore[valid-type]
Platform = Literal[tuple(KNOWN_PLATFORMS)]  # type: ignore[valid-type]
RelationshipStatus = Literal[tuple(KNOWN_RELATIONSHIP_STATUSES)]  # type: ignore[valid-type]


class PredictRequest(BaseModel):
    Age: int = Field(ge=10, le=100, description="Respondent age in years.")
    Gender: Gender
    Academic_Level: AcademicLevel
    Country: str = Field(min_length=2, max_length=60)
    Avg_Daily_Usage_Hours: float = Field(ge=0, le=24)
    Most_Used_Platform: Platform
    Sleep_Hours_Per_Night: float = Field(
        gt=0, le=24, description="Must be > 0 (see module docstring)."
    )
    Relationship_Status: RelationshipStatus
    Conflicts_Over_Social_Media: int = Field(ge=0, le=20)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 20,
                "Gender": "Female",
                "Academic_Level": "Undergraduate",
                "Country": "Pakistan",
                "Avg_Daily_Usage_Hours": 5.2,
                "Most_Used_Platform": "Instagram",
                "Sleep_Hours_Per_Night": 6.0,
                "Relationship_Status": "Single",
                "Conflicts_Over_Social_Media": 3,
            }
        }
    }


class FactorOut(BaseModel):
    feature: str
    pretty_name: str
    shap_value: float
    direction: Literal["raising", "lowering"]


class PredictResponse(BaseModel):
    predicted_score: float
    risk_tier: Literal["low_risk", "medium_risk", "high_risk"]
    confidence_interval_68pct: list[float]
    top_factors: list[FactorOut]
    explanation_sentence: str
    recommendations: list[str]
    model_name: str
    model_version: str
    disclaimer: str = (
        "This is a wellbeing indicator from a portfolio ML model, not a "
        "clinical diagnosis. If you're struggling, please talk to a "
        "counselor or another trusted person."
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    detail: str | None = None


# --- Phase 8 (Section 19/20): auth + check-in persistence ---
#
# POST /checkins reuses `PredictRequest` directly as its request body
# (see src/api/checkins.py) — the raw fields being submitted are
# identical, so a second near-duplicate schema would be exactly the
# kind of hand-rolled translation layer Section 18 already warned
# against for the encoder (see this file's module docstring).


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)


class RegisterResponse(BaseModel):
    email: str
    api_key: str
    note: str = (
        "Store this key — it is shown only once and is required as the "
        "X-API-Key header on every /checkins request."
    )


class CheckInSummary(BaseModel):
    """GET /checkins (list): one row per past check-in, without the
    full SHAP factor/explanation detail (that's what GET
    /checkins/{id} is for)."""

    id: int
    created_at: datetime
    predicted_score: float
    risk_tier: Literal["low_risk", "medium_risk", "high_risk"]


class CheckInDetail(BaseModel):
    """Returned by both POST /checkins (right after submission) and
    GET /checkins/{id} (retrieved later) — same shape either way, so
    a client can't tell whether it's looking at a fresh prediction or
    a persisted one, per Section 20's "show the same explanation
    again later without recomputing SHAP"."""

    id: int
    created_at: datetime
    predicted_score: float
    risk_tier: Literal["low_risk", "medium_risk", "high_risk"]
    confidence_interval_68pct: list[float]
    top_factors: list[FactorOut]
    explanation_sentence: str
    recommendations: list[str]
    model_name: str
    model_version: str
    disclaimer: str = (
        "This is a wellbeing indicator from a portfolio ML model, not a "
        "clinical diagnosis. If you're struggling, please talk to a "
        "counselor or another trusted person."
    )
