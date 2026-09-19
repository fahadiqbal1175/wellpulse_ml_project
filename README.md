# WellPulse: ML-Engineered Student Wellbeing Check-In

WellPulse predicts a continuous student wellbeing/mental-health score
(1–10) from behavioral and demographic survey data: social-media
usage, sleep, platform, academic level, relationship status, and
social-media conflict — and serves it through a fully tracked,
registered, tested, containerized, deployed, and monitored ML
pipeline. It's a project built to demonstrate end-to-end ML
engineering, not a clinical or diagnostic tool.

**Live app:** https://wellpulse.onrender.com
**API docs:** https://wellpulse.onrender.com/docs

Built phase-by-phase against a private technical specification
(`wellpulse-ml-blueprint-v2.md`, kept outside this repo) the
sections it references throughout `docs/` refer to that spec.

![CI](https://github.com/fahadiqbal1175/wellpulse_ml_project/actions/workflows/ci.yml/badge.svg)
![Drift check](https://github.com/fahadiqbal1175/wellpulse_ml_project/actions/workflows/drift_check.yml/badge.svg)

---

## Project status — all 14 phases complete

| # | Phase | Summary |
|---|---|---|
| 0 | Problem & dataset validation | Dataset confirmed: 705 rows, 13 columns, 0 missing/duplicates, 110 countries. |
| 1 | Data pipeline | Ingest + Pandera schema contract + content-hash versioning. |
| 2 | EDA & feature engineering | Leakage check excluded `Addicted_Score`; engineered features + train-only categorical encoding. |
| 3 | Baseline models | 7 model families compared; country-holdout generalization gap measured. |
| 4 | Advanced modeling & tuning | 9 families (+LightGBM/XGBoost); tuning did **not** beat the untuned baseline — reported as-is. |
| 5 | Final evaluation & explainability | Test-set evaluation, risk-tier report, error analysis, SHAP explainability. **Random Forest (tuned)** selected. |
| 6 | MLflow & model registry | Run tracking, model registration, `Production` alias promotion. |
| 7 | Inference API | FastAPI `POST /api/v1/predict` — real score, CI, SHAP factors, recommendation. |
| 8 | Database & application layer | Per-user API-key auth, SQLite-backed `/auth/register` + `/checkins`. |
| 9 | Frontend | Plain HTML/CSS/JS check-in UI, served as static files by FastAPI. |
| 10 | Dockerization | Multi-stage image (trainer → runtime); SQLite → Postgres migration. |
| 11 | CI/CD | GitHub Actions: lint, test, build, smoke-test on every push. |
| 12 | Deployment | Live on Render (web service + managed Postgres). |
| 13 | Monitoring & drift | Weekly PSI-based drift check over live check-in traffic via GitHub Actions. |


Full write-ups for each phase decisions made, numbers, limitations
live in [`docs/`](docs/).

---

## Architecture

```
Kaggle CSV
   │  (Phase 1: ingest + Pandera schema contract)
   ▼
data/raw/  ──►  Feature engineering + encoding (Phase 2)
                     │
                     ▼
        7→9 model families, tuned, evaluated on held-out test (Phase 3–5)
                     │
                     ▼
        MLflow tracking + registry, Production alias (Phase 6)
                     │
                     ▼
        FastAPI inference service (Phase 7)  ──►  SQLite/Postgres (Phase 8)
                     │
                     ▼
        Static HTML/CSS/JS frontend, served by FastAPI (Phase 9)
                     │
                     ▼
        Docker multi-stage image (Phase 10) ──► GitHub Actions CI (Phase 11)
                     │
                     ▼
        Render web service + managed Postgres (Phase 12)
                     │
                     ▼
        Weekly PSI drift check over live traffic (Phase 13)
```

## Tech stack

- **Data/ML:** pandas, Pandera, scikit-learn, LightGBM, XGBoost, SHAP
- **Experiment tracking:** MLflow (tracking + model registry)
- **API:** FastAPI, Pydantic, SQLAlchemy 2.0
- **DB:** SQLite (dev) / PostgreSQL (Docker & production)
- **Frontend:** vanilla HTML/CSS/JS, no build step, no framework
- **Infra:** Docker (multi-stage build), GitHub Actions, Render

---

## Dataset

Primary source: **Students' Social Media Addiction** (Kaggle,
[`adilshamim8/social-media-addiction-vs-relationships`](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships)),
705 rows, 13 columns, spanning 110 countries. Kaggle requires
authentication to fetch programmatically, so `data/raw/` here is
sourced from a GitHub mirror with matching column names and row count
(`rizanpradiya/Analyzing-Social-Media-Addiction-Among-Students`). Point
`make data SOURCE=...` at your own download if you'd rather pull it
from Kaggle directly.

`Addicted_Score` and `Affects_Academic_Performance` are **excluded**
from the model's feature set — both were found to explain most of the
target's variance on their own (89.3% and 65.4% respectively), a
near-tautological relationship rather than a genuine predictive
signal. Full reasoning in [`docs/FEATURES.md`](docs/FEATURES.md).

---

## Model

**Final model: Random Forest (tuned)** — selected over an untuned
Decision Tree on the test fold (MAE 0.2356 vs 0.2358, but Random
Forest wins clearly on RMSE and R²: 0.461/0.829 vs 0.505/0.796).
Neither LightGBM nor XGBoost beat either candidate, and hyperparameter
tuning did not beat the untuned baseline on validation — both
documented honestly rather than re-run until they "worked"
([`docs/PHASE4_TUNING.md`](docs/PHASE4_TUNING.md),
[`docs/PHASE5_EVALUATION.md`](docs/PHASE5_EVALUATION.md)).

**Known limitations, stated plainly:**
- High-risk recall is 0.5, half of true high-risk students in the
  test set are predicted into `medium_risk` instead. Precision on
  `high_risk` is 1.0 (no false alarms), but this is the single most
  important caveat if this project is ever read as more than a
  portfolio piece.
- Countries with very few rows (e.g. New Zealand, 8 rows total) are
  bucketed into "Other" by the categorical encoder and the model
  cannot distinguish them from other rare-country respondents
  measured directly as a ~5x higher error on New Zealand's test rows.
- A Random Forest's error roughly doubles on countries never seen
  during training (0.16 → 0.39 MAE) a real, measured generalization
  gap, not a hypothetical one.
- Confidence intervals are a residual-spread proxy (± 1 std, measured
  on validation), not a statistically calibrated prediction interval.

---

## Getting started (local, no Docker)

```bash
pip install -r requirements.txt

# Re-validate the dataset already at data/raw/, or ingest a fresh copy:
make data                              # SOURCE=~/Downloads/students.csv to ingest fresh

# Rebuild the EDA notebook (optional — already committed with fresh output):
make eda

# Train, tune, and evaluate:
make baseline       # 7 model families
make advanced       # +LightGBM/XGBoost (9 total)
make tune           # randomized search on the top-2 by val MAE
make phase5         # final test-set evaluation + error analysis + SHAP

# Register the final model into a local MLflow registry (writes mlflow.db, gitignored):
make register-model

# Serve the API + frontend at http://localhost:8000
make serve-api
```

`make register-model` must run at least once locally or via Docker
before the API has a `Production` model to load.

## Running the full stack with Docker

```bash
make docker-up      # builds if needed, waits for Postgres, serves at http://localhost:8000
make docker-down    # stops both containers (check-in history survives; add -v to wipe it)
make docker-logs
```

The image is a two-stage build: a **trainer** stage registers the
already-trained, already-committed model into a fresh MLflow store
rooted at the image's own filesystem (fixing a real portability bug
where MLflow bakes in an absolute artifact path, see
[`docs/PHASE10_DOCKERIZATION.md`](docs/PHASE10_DOCKERIZATION.md)), and
a **runtime** stage that actually serves traffic, using only
`requirements-serve.txt` (no LightGBM/XGBoost/notebook tooling in the
production image).

## Testing

```bash
make test     # 71/71 passing
```

Covers data validation (including deliberately corrupted inputs),
feature engineering, splits, baseline/tuning, evaluation, the MLflow
registry, the inference API (including a real prediction against the
real Production model), auth + check-in persistence, drift math, and
static-frontend serving.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`: lint
(ruff) → register the model into a throwaway registry → full test
suite → `docker compose up --build` → smoke-test `/health` and one
real `/api/v1/predict` call against the running container → tear down.

## Deployment

Hosted on **Render** (free tier): one Docker web service (built
straight from the repo's existing `Dockerfile`, no changes needed) +
one managed Postgres instance, connected over Render's private
network. Verified against the live URL, not just locally, see
[`docs/PHASE12_DEPLOYMENT.md`](docs/PHASE12_DEPLOYMENT.md) for the
free-tier limitations (cold starts, 30-day Postgres expiry, no
staging environment).

## Monitoring & drift

`.github/workflows/drift_check.yml` runs weekly (and on manual
dispatch): pulls the last 200 check-ins from the production database,
compares each feature's and the predicted score's distribution against
a committed training-time reference using **Population Stability Index
(PSI)**, and commits the report back to
[`reports/monitoring/`](reports/monitoring/) either way. The workflow
fails (red ✗) when the verdict is `"significant"`. Verified for real:
a deliberately shifted synthetic batch was seeded against the live
production database and confirmed to trip the check. Full design
rationale — why PSI alone, why a 200-check-in rolling window instead
of a day-based one, why 5 bins instead of 10, in
[`docs/PHASE13_MONITORING_DRIFT.md`](docs/PHASE13_MONITORING_DRIFT.md).

This only measures feature/prediction drift, never accuracy: no
ground-truth label is ever collected on a production check-in, so
there's nothing to score the model's real-world correctness against.

## API reference

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Liveness + whether the Production model is loaded |
| `POST /api/v1/predict` | none | Free "try it" endpoint — score, CI, SHAP factors, recommendation. Not persisted. |
| `POST /auth/register` | none | Register with an email, get a per-user API key (shown once) |
| `POST /checkins` | API key | Submit a check-in — persisted, same response shape as `/predict` |
| `GET /checkins` | API key | List your own check-in history, newest first |
| `GET /checkins/{id}` | API key | Re-fetch one of your own past check-ins (404 if not yours) |

Interactive docs at `/docs` (Swagger UI) once the API is running.

## Repository layout

```
src/
  data/          Ingestion, Pandera schema, train/val/test + country-holdout splits
  features/      Deterministic feature engineering + train-only categorical encoding
  models/        Baseline/advanced leaderboards, hyperparameter tuning
  evaluation/    Final test-set evaluation, error analysis, SHAP explainability
  experiments/   Experiment logging, MLflow registration
  api/           FastAPI app, schemas, auth, check-ins, inference service, recommendations
  monitoring/    Reference distribution, PSI drift math, scheduled drift-check job
static/          Frontend (index.html, style.css, app.js)
notebooks/       Generated + executed EDA notebook
docs/            Per-phase write-ups: decisions, numbers, limitations
reports/         Leaderboards, evaluation reports, figures, monitoring reports
tests/           71 tests across every phase above
.github/workflows/  ci.yml (build/test/smoke-test), drift_check.yml (weekly monitoring)
```

