"""
Phase 13 — Monitoring & Drift: seeds N deliberately-shifted synthetic
check-ins directly into the DB, for exercising the drift job without
waiting on real traffic. Milestone ML-9's verify criterion: "a
deliberately shifted synthetic request batch triggers a drift flag."

Run against local SQLite/Postgres for a dry run, or point DATABASE_URL
at the Render Postgres's External Database URL to prove the milestone
against the actual production database. Manual/one-off tool, not
wired into CI.
"""
from __future__ import annotations

import argparse
import random

from src.api.db import SessionLocal, init_db
from src.api.db_models import CheckIn, Prediction, User, generate_api_key
from src.api.inference import ModelService

SYNTHETIC_USER_EMAIL = "drift-test@wellpulse.local"


def _get_or_create_synthetic_user(db) -> User:
    user = db.query(User).filter_by(email=SYNTHETIC_USER_EMAIL).first()
    if user is None:
        user = User(email=SYNTHETIC_USER_EMAIL, api_key=generate_api_key())
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def _shifted_raw_row() -> dict:
    """Deliberately far outside the training distribution on several
    features at once (sleep, usage, age, an unseen country/platform)
    so the drift job has no ambiguity to resolve."""
    return {
        "Age": random.randint(40, 60),
        "Gender": random.choice(["Male", "Female"]),
        "Academic_Level": "Graduate",
        "Country": "Freedonia",  # not in training data at all
        "Avg_Daily_Usage_Hours": round(random.uniform(9.5, 12.0), 1),
        "Most_Used_Platform": "LINE",  # rare in training
        "Sleep_Hours_Per_Night": round(random.uniform(0.5, 2.0), 1),
        "Relationship_Status": "Complicated",
        "Conflicts_Over_Social_Media": random.randint(15, 20),
    }


def seed(n: int) -> None:
    init_db()
    service = ModelService()
    service.load()
    if not service.is_ready:
        raise RuntimeError(f"Could not load the Production model: {service.load_error}")

    db = SessionLocal()
    try:
        user = _get_or_create_synthetic_user(db)
        for _ in range(n):
            raw = _shifted_raw_row()
            result = service.predict_one(raw)

            check_in = CheckIn(user_id=user.id, **{k.lower(): v for k, v in raw.items()})
            db.add(check_in)
            db.flush()  # assigns check_in.id

            db.add(Prediction(
                check_in_id=check_in.id,
                predicted_score=result["predicted_score"],
                risk_tier=result["risk_tier"],
                model_name=result["model_name"],
                model_version=result["model_version"],
                confidence_interval_low=result["confidence_interval_68pct"][0],
                confidence_interval_high=result["confidence_interval_68pct"][1],
            ))
        db.commit()
        print(f"Seeded {n} synthetic, deliberately-shifted check-ins as {SYNTHETIC_USER_EMAIL}.")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    seed(args.n)