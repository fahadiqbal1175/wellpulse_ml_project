"""
Phase 8 — SQLAlchemy ORM models (Section 20).

Section 20 lists three tables — `check_ins`, `predictions`,
`prediction_explanations` — and this file builds exactly those three,
plus one addition:

  - `users` — NOT one of Section 20's three tables. It's a necessary
    consequence of the resolved auth decision ("build simple auth
    now"): Section 19's "list own history" needs an identity for
    "own" to mean anything. Flagged here explicitly (same practice as
    Phase 7's recommendations.py addition) rather than silently
    folded in as if Section 20 had asked for it. Deliberately minimal
    per Section 19's own wording ("a single API key... enough...
    without building a full identity system"): email + a
    server-generated API key, no password hashing, no JWT, no
    sessions.

Explicitly NOT built (Section 20's own exclusions — still correct
here, nothing about the auth decision changes them):
  - `model_versions` (MLflow's registry already owns this)
  - cohort / consent / counselor tables
  - free-text notes
"""
from __future__ import annotations

import secrets
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def generate_api_key() -> str:
    """URL-safe, 32 bytes of entropy — plenty for a portfolio-scale
    per-user credential; not intended to double as a password."""
    return secrets.token_urlsafe(32)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    api_key: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, default=generate_api_key
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    check_ins: Mapped[list["CheckIn"]] = relationship(back_populates="user")


class CheckIn(Base):
    """Section 20: "the raw submitted features (needed to
    reproduce/audit any prediction...)". Columns mirror
    `PredictRequest`'s 9 fields (src/api/schemas.py) exactly — the
    same raw shape the model actually consumes, not a reinterpreted
    version of it."""

    __tablename__ = "check_ins"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[str] = mapped_column(String(20))
    academic_level: Mapped[str] = mapped_column(String(30))
    country: Mapped[str] = mapped_column(String(60))
    avg_daily_usage_hours: Mapped[float] = mapped_column(Float)
    most_used_platform: Mapped[str] = mapped_column(String(30))
    sleep_hours_per_night: Mapped[float] = mapped_column(Float)
    relationship_status: Mapped[str] = mapped_column(String(30))
    conflicts_over_social_media: Mapped[int] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    user: Mapped["User"] = relationship(back_populates="check_ins")
    prediction: Mapped["Prediction"] = relationship(
        back_populates="check_in", uselist=False, cascade="all, delete-orphan"
    )


class Prediction(Base):
    """Section 20: "score, model_version, confidence interval,
    timestamp (needed for monitoring and future retraining data)"."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    check_in_id: Mapped[int] = mapped_column(
        ForeignKey("check_ins.id"), unique=True, index=True
    )

    predicted_score: Mapped[float] = mapped_column(Float)
    risk_tier: Mapped[str] = mapped_column(String(20))
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(100))
    confidence_interval_low: Mapped[float] = mapped_column(Float)
    confidence_interval_high: Mapped[float] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    check_in: Mapped["CheckIn"] = relationship(back_populates="prediction")
    explanation: Mapped["PredictionExplanation"] = relationship(
        back_populates="prediction", uselist=False, cascade="all, delete-orphan"
    )


class PredictionExplanation(Base):
    """Section 20: "top-factor SHAP output per prediction (needed to
    show the same explanation again later without recomputing
    SHAP)". `recommendations` is stored for the identical reason —
    Section 15's rule-based lookup was already flagged as a Phase 7
    addition beyond the narrow ML-6 milestone; storing its output
    here (so a past check-in can show the same recommendations again
    without recomputing them) is that same addition, extended for the
    same stated reason, not new scope."""

    __tablename__ = "prediction_explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id"), unique=True, index=True
    )

    top_factors: Mapped[list] = mapped_column(JSON)
    explanation_sentence: Mapped[str] = mapped_column(String(1000))
    recommendations: Mapped[list] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    prediction: Mapped["Prediction"] = relationship(back_populates="explanation")
