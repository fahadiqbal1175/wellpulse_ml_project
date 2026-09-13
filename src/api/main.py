"""
Phase 7/8 — Inference API + Database/Application Layer (Section
18/19/20), Milestones ML-6 and Phase 8's "a submitted check-in
round-trips through the DB".

Endpoints:
    GET  /health              liveness/readiness (Section 19's `health` router)
    POST /api/v1/predict      score + confidence interval + SHAP factors +
                              recommendations — free, unauthenticated, not persisted
    POST /auth/register       Phase 8: email in, per-user API key out (shown once)
    POST /checkins            Phase 8: authenticated, persisted version of /predict
    GET  /checkins            Phase 8: the caller's own check-in history
    GET  /checkins/{id}       Phase 8: one past check-in, full detail
    GET  /, /style.css, /app.js, ...
                              Phase 9: the frontend (Section 21) — a static
                              HTML/JS/CSS bundle in `static/`, mounted at "/"
                              AFTER every API route below so explicit paths
                              (e.g. /health) always win; StaticFiles only
                              ever serves what those routes don't claim.
                              Plain HTML/JS, not React — Section 34/35 frame
                              the frontend as MVP-minimal (~5% of project
                              effort), and serving it from this same FastAPI
                              process avoids CORS/build tooling entirely.

Deliberately NOT included:
    - An admin-triggered model-reload endpoint — Section 18 mentions
      this as one option for picking up a registry-alias change, but
      it's optional ("checked periodically OR via an admin-triggered
      reload endpoint") and outside any phase's roadmap wording so
      far. Restart the process to pick up a new Production alias.
    - Alembic/migration framework — see src/api/db.py's docstring.

Run locally:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src.api import auth, checkins
from src.api.db import init_db
from src.api.inference import ModelService, ModelUnavailableError
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once per process (Section 18: "never reloaded per
    # request"). A failed load does NOT prevent the app from starting
    # — it leaves the service in a not-ready state so /health and
    # /predict can report a clear 503 instead of the process refusing
    # to boot at all.
    service = ModelService()
    service.load()
    app.state.model_service = service

    # Phase 8: creates users/check_ins/predictions/prediction_explanations
    # if they don't exist yet. A missing/misconfigured DB fails startup
    # loudly here (unlike the model, there's no sensible "degraded"
    # mode for an app with no database at all).
    init_db()

    yield


app = FastAPI(
    title="WellPulse Inference API",
    description="Predicts a student wellbeing score from behavioral/demographic survey data.",
    version="0.8.0",
    lifespan=lifespan,
)

app.include_router(auth.router)
app.include_router(checkins.router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service: ModelService = app.state.model_service
    if service.is_ready:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=service.model_name,
            model_version=service.model_version,
        )
    return HealthResponse(
        status="unavailable",
        model_loaded=False,
        detail=service.load_error,
    )


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    # Pydantic has already run full validation (ranges, known
    # categories) by the time this function body executes — a
    # malformed request never reaches this line at all, it gets a 422
    # straight from FastAPI (Section 18: "malformed input -> 422
    # before the model is ever touched").
    service: ModelService = app.state.model_service
    try:
        result = service.predict_one(request.model_dump())
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - unexpected pipeline failure
        logger.exception("Prediction failed unexpectedly")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc
    return PredictResponse(**result)


# Phase 9 (Section 21): mounted LAST and at "/" so every API route
# above still wins on an exact match — Starlette tries routes in
# registration order, and this Mount only ever catches what nothing
# above it claimed (e.g. GET /, GET /style.css, GET /app.js).
# check_dir=True (the default) means this raises loudly at import
# time if static/ is ever missing, rather than silently 404ing every
# frontend request.
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")
