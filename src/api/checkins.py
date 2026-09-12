"""
Phase 8 — `/checkins` (Section 19/20, roadmap Phase 8: "persistence
for check-ins/predictions/explanations... a submitted check-in
round-trips through the DB").

POST /checkins calls `ModelService.predict_one()` — the exact same
Phase 7 prediction logic `/api/v1/predict` uses — and never
reimplements any part of it (feature prep, SHAP, confidence interval,
risk tier, recommendations all come from there unchanged). It then
persists the check-in + prediction + explanation as one unit.

`/api/v1/predict` (Phase 7) is left untouched: it stays the free,
unauthenticated, non-persisting "try it" endpoint. `/checkins` is the
new authenticated, persisted path — "this submission is really me,
and I want it saved."
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.auth import get_current_user
from src.api.db import get_db
from src.api.db_models import CheckIn, Prediction, PredictionExplanation, User
from src.api.inference import ModelService, ModelUnavailableError
from src.api.schemas import CheckInDetail, CheckInSummary, PredictRequest

router = APIRouter()


def _to_detail(check_in: CheckIn) -> CheckInDetail:
    prediction = check_in.prediction
    explanation = prediction.explanation
    return CheckInDetail(
        id=check_in.id,
        created_at=check_in.created_at,
        predicted_score=prediction.predicted_score,
        risk_tier=prediction.risk_tier,
        confidence_interval_68pct=[
            prediction.confidence_interval_low,
            prediction.confidence_interval_high,
        ],
        top_factors=explanation.top_factors,
        explanation_sentence=explanation.explanation_sentence,
        recommendations=explanation.recommendations,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
    )


@router.post("/checkins", response_model=CheckInDetail, status_code=201)
def submit_checkin(
    request: PredictRequest,
    http_request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CheckInDetail:
    service: ModelService = http_request.app.state.model_service
    try:
        result = service.predict_one(request.model_dump())
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - unexpected pipeline failure
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    check_in = CheckIn(
        user_id=user.id,
        age=request.Age,
        gender=request.Gender,
        academic_level=request.Academic_Level,
        country=request.Country,
        avg_daily_usage_hours=request.Avg_Daily_Usage_Hours,
        most_used_platform=request.Most_Used_Platform,
        sleep_hours_per_night=request.Sleep_Hours_Per_Night,
        relationship_status=request.Relationship_Status,
        conflicts_over_social_media=request.Conflicts_Over_Social_Media,
    )
    db.add(check_in)
    db.flush()  # assigns check_in.id without committing yet

    prediction = Prediction(
        check_in_id=check_in.id,
        predicted_score=result["predicted_score"],
        risk_tier=result["risk_tier"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        confidence_interval_low=result["confidence_interval_68pct"][0],
        confidence_interval_high=result["confidence_interval_68pct"][1],
    )
    db.add(prediction)
    db.flush()  # assigns prediction.id

    explanation = PredictionExplanation(
        prediction_id=prediction.id,
        top_factors=result["top_factors"],
        explanation_sentence=result["explanation_sentence"],
        recommendations=result["recommendations"],
    )
    db.add(explanation)
    db.commit()
    db.refresh(check_in)

    return _to_detail(check_in)


@router.get("/checkins", response_model=list[CheckInSummary])
def list_checkins(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[CheckInSummary]:
    check_ins = (
        db.execute(
            select(CheckIn)
            .where(CheckIn.user_id == user.id)
            .order_by(CheckIn.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [
        CheckInSummary(
            id=c.id,
            created_at=c.created_at,
            predicted_score=c.prediction.predicted_score,
            risk_tier=c.prediction.risk_tier,
        )
        for c in check_ins
    ]


@router.get("/checkins/{checkin_id}", response_model=CheckInDetail)
def get_checkin(
    checkin_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CheckInDetail:
    check_in = db.get(CheckIn, checkin_id)
    # 404 (not 403) whether it doesn't exist or belongs to someone
    # else — doesn't confirm/deny another user's check-in ID exists.
    if check_in is None or check_in.user_id != user.id:
        raise HTTPException(status_code=404, detail="Check-in not found.")
    return _to_detail(check_in)
