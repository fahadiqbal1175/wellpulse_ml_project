# Phase 7 — Inference API

## Objective

FastAPI service exposing the Production model (Section 18/19,
Milestone ML-6): `POST /api/v1/predict` returns a real score,
confidence interval, and SHAP explanation from a real HTTP request.

## Scope decisions made before coding

Two things the blueprint left ambiguous for this phase, resolved with
Fahad before writing any code:

1. **`/checkins`** — Section 19 suggests it can be folded into
   `/predict` ("a check-in *is* a prediction request"), but DB
   persistence is explicitly Phase 8 (Section 20). **Decision: skip
   `/checkins` entirely until Phase 8 exists** — no endpoint with
   nothing to persist to.
2. **Repo structure** — the blueprint's Section 32 layout wants
   `src/inference/` + a top-level `api/`. This repo already deviates
   from the blueprint's `src/training/` in favor of `src/models/`.
   **Decision: keep it flatter — one `src/api/` folder** for routes,
   schemas, and inference logic together, rather than adding the
   blueprint's exact two-folder split.

A third item came up during implementation, not in the original two
questions: Section 18's response pipeline diagram places a
**"Recommendation lookup"** step between SHAP and the response, and
Section 15 designs it to be cheap and self-contained. It's included
here (`src/api/recommendations.py`) but flagged as an addition beyond
Milestone ML-6's narrow "score + SHAP factors" wording — easy to
strip out if you'd rather defer it to Phase 8.

## Environment note: stale MLflow artifact paths

The shipped `mlflow.db`/`mlruns/` had absolute artifact paths baked in
from wherever they were first created — `mlflow.sklearn.load_model()`
failed until the whole Phase 3→6 pipeline was rerun fresh in this
sandbox (`baseline → advanced → tune → evaluate → error-analysis →
explain → register-model`). Every number reproduced exactly (untuned
Decision Tree 0.1509, tuned RF val MAE 0.1704, test MAE 0.2356,
Production v1). This is a real portability gap in a local
filesystem-backed MLflow artifact store, not a code bug — worth
solving properly before Phase 10 (Docker), where the same issue will
otherwise resurface.

## What was built

```
src/api/
  schemas.py          Pydantic PredictRequest/PredictResponse/HealthResponse
  recommendations.py  rule-based lookup (Section 15) — only touches the ranked factor list
  inference.py         ModelService: loads Production model + encoder once, predicts + explains
  main.py               FastAPI app, lifespan-managed startup, GET /health, POST /api/v1/predict
tests/test_api.py       11 tests, including a real prediction against the actual Production model
```

**Field naming in `PredictRequest`:** fields reuse the exact raw
dataset column names (`Age`, `Gender`, `Academic_Level`, ...) instead
of idiomatic snake_case, specifically to avoid a hand-rolled
snake_case → PascalCase mapping layer sitting between the API and
`add_engineered_features`/`encoder.transform` — exactly the kind of
manual translation Section 18 flags as the most common real-world
source of training/serving skew.

**Encoder sourcing:** `mlflow.sklearn.log_model()` logs only the raw
sklearn estimator, not a pyfunc wrapper with the encoder bundled — so
despite Section 18's "loaded from the registry alongside the model"
wording, the registry alone has no encoder to load. The encoder comes
from `models/final_model.joblib` instead, which `register_final_model.py`
guarantees is the same `(model, encoder)` pair that got logged in the
same run.

**Validation tightening:** `Sleep_Hours_Per_Night` is `gt=0` in the API
(the raw ingestion schema in `src/data/schema.py` allows `ge=0`) — a
literal 0 divides-by-zero in `usage_to_sleep_ratio`, a value the
training data never actually contained.

**Reused, not reimplemented:** `add_engineered_features`,
`CategoricalFeatureEncoder.transform`, `compute_shap_values`,
`top_n_factors`, `generate_explanation_sentence`, `CONFIDENCE_Z`,
`compute_residual_std_on_val`, `compute_risk_tier` — all imported
directly from Phase 2/5 modules.

## Verification

- Full test suite: **49/49 passed** (38 existing + 11 new).
- Live `uvicorn` process, real HTTP requests:
  - `GET /health` → `{"status":"ok","model_loaded":true,"model_name":"Random Forest (tuned)","model_version":"wellpulse_final_model:1"}`
  - `POST /api/v1/predict` with a real payload → real score (5.944),
    medium_risk tier, 68% CI, 3 ranked SHAP factors, a recommendation,
    and the disclaimer.
  - Malformed input (`Age: 5`) → `422`, model never touched (asserted
    in `test_predict_malformed_input_returns_422_before_touching_model`
    via monkeypatching `predict_one` to raise if called at all).
  - `/docs` and `/openapi.json` both serve correctly (FastAPI's
    auto-generated interactive docs).
- Missing-model path (`test_predict_missing_model_returns_503`,
  `test_health_reports_unavailable_when_model_missing`): simulated via
  monkeypatching `ModelService.model = None` rather than an empty
  registry, since `tests/test_api.py` intentionally overrides the
  project's autouse MLflow-isolation fixture to test against the real
  registry.

## Not built this phase (by design)

- `/checkins` (Phase 8 — needs the DB to mean anything).
- An admin-triggered model-reload endpoint (Section 18 lists this as
  optional; restart the process to pick up a new Production alias).
- Prediction request logging (Section 18 ties this to the DB, i.e.
  Phase 8).
