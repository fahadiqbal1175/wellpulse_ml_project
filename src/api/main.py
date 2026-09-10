"""
Phase 7 — Inference API (Section 18/19), Milestone ML-6.

Endpoints:
    GET  /health              liveness/readiness (Section 19's `health` router)
    POST /api/v1/predict      score + confidence interval + SHAP factors + recommendations

Deliberately NOT included (per this phase's scoping decisions):
    - /checkins — Section 20 (DB persistence) is Phase 8; building a
      /checkins endpoint with nothing to persist to was decided against
      for this phase.
    - An admin-triggered model-reload endpoint — Section 18 mentions
      this as one option for picking up a registry-alias change, but
      it's optional ("checked periodically OR via an admin-triggered
      reload endpoint") and outside Phase 7's roadmap wording. Restart
      the process to pick up a new Production alias for now.

Run locally:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

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
    yield


app = FastAPI(
    title="WellPulse Inference API",
    description="Predicts a student wellbeing score from behavioral/demographic survey data.",
    version="0.7.0",
    lifespan=lifespan,
)


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
