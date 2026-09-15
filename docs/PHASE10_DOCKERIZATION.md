# Phase 10 — Dockerization

## Objective

Dockerfile(s) + compose per Section 22. Verify: `docker-compose up`
serves the full stack locally (Milestone ML-7).

## Sandbox constraint, stated up front

This sandbox does not have Docker installed (`docker: not found`), so
a literal `docker-compose up`/`docker build` could not be run here —
unlike every previous phase's "actually run it, don't simulate"
standard. Verification below is split accordingly:

- **What Docker itself would do** (multi-stage image build, container
  orchestration) — verified by careful hand-construction/review of
  the Dockerfile/compose file, not an actual build.
- **What Docker would be wrapping** — the SQLite→Postgres migration,
  and the exact runtime dependency set the serving image installs —
  verified for real: a real local Postgres (installed via `apt`, an
  allowed network domain here) and a real isolated Python environment
  containing only `requirements-serve.txt`, both driven with a live
  `uvicorn` process and real HTTP requests.

This mirrors what Fahad and I agreed before writing any code.

## Scope decision made before coding

**Question:** include a separate `mlflow` service in
`docker-compose.yml`, or keep the local file-based tracking store
Section 22 allows as an alternative?

**Decision: keep the file-based store, baked into the image.** The
Dockerfile's trainer stage (below) already produces a fully working
`mlflow.db` + `mlruns/` with a Production alias set — the `api`
container never needs a live MLflow server to serve predictions. For
ad hoc run-history browsing, `make mlflow-ui` still works locally
exactly as it did in Phase 6; compose doesn't need to duplicate it.

## The real design problem this phase surfaced

Every phase's hand-off note so far has hit the same MLflow issue:
`mlflow.db` bakes in an **absolute** artifact path
(`ARTIFACT_ROOT.resolve().as_uri()`) at experiment-creation time,
which only resolves correctly on the machine that created it — hence
"regenerate mlflow.db/mlruns fresh in your sandbox" every phase.
Docker actually gives a permanent fix, not just another workaround:
**build the registry inside the image, at the same path the image
will run at.** Concretely, the Dockerfile's two stages both use
`WORKDIR /app`, so whatever machine or CI runner runs `docker build`
stops mattering — the path baked into `mlflow.db` always resolves
inside that same image, because the container that reads it is the
same filesystem that wrote it.

## Second design finding: the "trainer" stage doesn't retrain anything

Initial assumption going in was that the image's build-time stage
would need to re-run the full chain (`ingest → baseline → advanced →
tune → evaluate → error-analysis → explain → register-model`), same
as every sandbox-refresh so far. Traced the actual import graph
instead of assuming, and confirmed `python -m
src.experiments.register_final_model`'s only real inputs are
`data/raw/students_social_media_addiction.csv` and
`models/phase4_best_model.joblib` (both already git-committed —
`.gitignore` only excludes `mlflow.db`/`mlruns`/`wellpulse_app.db`,
never `data/raw/` or `models/`). Verified directly: deleted
`mlflow.db`/`mlruns`, ran **only** `register_final_model.py` with no
prior `baseline`/`advanced`/`tune` step, and got an identical result
(test MAE 0.2356, Random Forest (tuned) promoted to Production) —
confirmed against the full 63-test suite passing afterward too.

This matters beyond just build speed: re-running the full
training/tuning chain on every `docker build` would mean Phase 11's
CI retrains a model on every code push, which Section 23 explicitly
rules out ("retraining is a separate, scheduled pipeline...
conflating them would mean every typo fix retrains a model
unnecessarily"). The trainer stage's real job is **registration**, not
training — it packages an already-trained, already-committed model
into a tracking store shaped correctly for this image.

## Third finding: the runtime image is heavier than Section 22's own framing suggests

Section 22 frames the serving image as one that "includes the loaded
model artifact." In this codebase, `ModelService.load()` does load
`models/final_model.joblib` once at startup — but it also calls
`compute_residual_std_on_val()` (`src/evaluation/explainability.py`),
which **on every process startup**: re-reads
`data/raw/students_social_media_addiction.csv`, re-validates it
through Pandera, rebuilds the train/val/test split, and **refits a
fresh `DecisionTreeRegressor`** (`build_final_eval_bundle()` in
`src/evaluation/final_evaluation.py`) to get the residual spread for
the 68% confidence interval. That's a Phase 5 design choice, not
something this phase changes — but it means the runtime image
genuinely needs `data/raw/`, **both** `models/phase4_best_model.joblib`
and `models/final_model.joblib`, and most of `src/` (`data`,
`features`, `models`, `evaluation`, not just `api`), not only a single
lightweight artifact file.

## What was built

```
Dockerfile               Multi-stage: trainer (registers the model into
                          a fresh MLflow store) -> runtime (serves it)
docker-compose.yml        api + db (Postgres) for local dev
.dockerignore              Trims build context; keeps host mlflow.db/
                          mlruns out of it (trainer stage regenerates
                          its own, deliberately not reusing the host's)
requirements-serve.txt    Pruned dependency set for BOTH Docker stages
requirements.txt           + psycopg2-binary (Phase 10), cross-referenced
                          with requirements-serve.txt
src/api/db.py              Docstring updated: migration now DONE, not
                          planned (no code changes — see below)
Makefile                   docker-build / docker-up / docker-down /
                          docker-logs targets (Linux/Mac/WSL convenience;
                          Windows/PowerShell users run the plain
                          `docker`/`docker compose` commands directly)
```

**`requirements-serve.txt` vs. `requirements.txt`:** verified, not
assumed. A fresh venv installed from only `requirements-serve.txt`
(fastapi, uvicorn, sqlalchemy, psycopg2-binary, pandas, pandera,
scikit-learn, joblib, shap, mlflow, matplotlib — no lightgbm, xgboost,
seaborn, nbformat, nbclient, or ipykernel) successfully imported
`src.api.main`, served a live `/health` + `/api/v1/predict` request
with a real SHAP explanation, and ran
`src.experiments.register_final_model` end to end. Confirms none of
the serving or registration code paths import the training-only
packages.

## Verification

**SQLite → Postgres migration (real Postgres, not mocked):**
- Installed Postgres 16 via `apt` in this sandbox; created a
  `wellpulse` role/database.
- `DATABASE_URL=postgresql://wellpulse:wellpulse@localhost/wellpulse`
  + `init_db()` → all 4 tables (`users`, `check_ins`, `predictions`,
  `prediction_explanations`) created, **zero changes** to
  `src/api/db.py` or `db_models.py` — confirms the Phase 8
  `DATABASE_URL`-first design worked exactly as intended.
- Live `uvicorn` process against that same Postgres database, driven
  with real HTTP requests: register → 201; duplicate email → 409;
  second user → 201; real check-in with real inference → 201
  (persisted, SHAP factors returned); list history → 200; get by ID →
  200; second user fetching the first user's check-in → 404; invalid
  API key → 401; malformed body → 422. Confirmed the rows actually
  landed in Postgres via `psql` directly (`SELECT` against `users`,
  `check_ins`, `predictions`).
- Full test suite: **63/63 passed**, unaffected by any of the above —
  `tests/conftest.py`/`test_checkins.py` already isolate each test
  onto its own temp-file SQLite database regardless of `DATABASE_URL`,
  by design from Phase 8.

**Runtime dependency footprint (isolated venv, not the full project
env):**
- Fresh venv, `pip install -r requirements-serve.txt` only.
- `python -c "import src.api.main"` → clean import.
- Live `uvicorn` process on that venv, real HTTP: `/health` → real
  Production model reported loaded; `POST /api/v1/predict` → real
  score (7.946), risk tier, 3 SHAP factors, real confidence interval —
  confirms `compute_residual_std_on_val()`'s full re-derivation path
  (Pandera validation, split rebuild, DecisionTreeRegressor refit)
  works with nothing beyond `requirements-serve.txt` installed.
- (An `InconsistentVersionWarning` from scikit-learn appeared here,
  because this ad hoc venv resolved a newer scikit-learn than the
  main sandbox environment's already-installed one — an artifact of
  testing two environments side by side, not a real issue: a single
  `docker build` installs each stage's `requirements-serve.txt` once,
  from the same index, so both stages land on the same resolved
  version. Still worth pinning exactly in both requirements files if
  this warning is ever seen from an actual container.)

**Dockerfile/compose (hand-reviewed, not built — see constraint
above):**
- Multi-stage build verified line-by-line against the two findings
  above: trainer stage installs `requirements-serve.txt`, copies only
  `src/`, `data/raw/`, `models/` (not `notebooks/`, `scripts/`,
  `tests/`, `docs/`, `reports/`), and runs exactly the one command
  (`register_final_model.py`) confirmed sufficient above.
- Runtime stage copies `src/`, `static/`, `data/raw/`, `models/`, and
  `mlflow.db`/`mlruns/` **from the trainer stage** (`COPY --from=
  trainer`) rather than rebuilding them a second time.
- `docker-compose.yml`'s `api` healthcheck uses plain `python` +
  `urllib` rather than `curl`/`wget`, since `python:3.12-slim` (the
  runtime stage's base image) has neither installed and adding one
  would work against Section 22's "keep the production image small"
  goal for a healthcheck alone.
- `depends_on: db: condition: service_healthy` + a `pg_isready`
  healthcheck on `db` — the `api` container's own `init_db()` call
  would otherwise race a still-starting Postgres.

## Not built this phase (by design)

- A live `mlflow` service in compose — see the resolved decision
  above.
- Re-running `baseline`/`advanced`/`tune` at image-build time — see
  the second design finding above; the CI implications land in
  Phase 11, but the reasoning not to bake retraining into every build
  starts here.
- A `HEALTHCHECK` instruction inside the Dockerfile itself —
  `docker-compose.yml`'s healthcheck covers it for local dev; adding a
  second, differently-configured one on the image would just be a
  second source of truth to keep in sync.
- Alembic/other migration tooling for the Postgres schema — unchanged
  from Phase 8's stated scope; `init_db()` still just creates tables
  that don't exist yet.
- An actual `docker build`/`docker-compose up` run — the sandbox
  constraint stated up front. Fahad's own machine (with Docker
  installed) is the first place this can be verified end-to-end
  exactly as written; the step-by-step guide flags where to watch for
  trouble.
