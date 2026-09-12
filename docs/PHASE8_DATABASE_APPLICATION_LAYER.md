# Phase 8 — Database/Application Layer

## Objective

Persistence for check-ins/predictions/explanations (Section 20;
roadmap Phase 8). Verify: a submitted check-in round-trips through
the DB.

## Scope decisions made before coding

Two things the blueprint left open for this phase, resolved with
Fahad before writing any code:

1. **DB choice** — Section 34 marks Postgres "Essential" but also
   lists SQLite as a simpler local-dev alternative, and Section 22's
   `docker-compose.yml` doesn't introduce Postgres until Phase 10.
   **Decision: SQLite now, Postgres at Phase 10.** `src/api/db.py`
   reads `DATABASE_URL` from the environment first, so the migration
   is a connection-string change — the ORM models use only portable
   SQLAlchemy types (no SQLite-specific columns).
2. **Minimal auth** — Section 19: "a single API key (or simple
   email/password login) is enough... without building a full
   identity system." **Decision: build it now**, as a *per-user* API
   key (not one shared key) — needed so `GET /checkins` ("list own
   history") has a real "own" to filter by. `POST /auth/register`
   takes only an email and returns a server-generated key, shown
   once. No password hashing, no JWT, no sessions.

A third item followed directly from decision 2, not from Section 20's
own text: a **`users` table**. Section 20 lists exactly three tables
(`check_ins`, `predictions`, `prediction_explanations`) and doesn't
mention one. It's added here because "own" history is meaningless
without an identity to own it — flagged the same way Phase 7 flagged
`recommendations.py` as an addition beyond the narrow milestone
wording, not silently folded in as if Section 20 had asked for it.

## Environment note: same MLflow lesson, applied again

The Phase 7 zip's `mlflow.db`/`mlruns/` had Fahad's Windows path baked
in (`No such artifact: ''` on load) — same failure mode as Phase 7's
own environment note, now doubly confirmed. Deleted both and reran
the full chain (`baseline → advanced → tune → evaluate →
error-analysis → explain → register-model`) fresh in this sandbox.
Every number reproduced exactly: untuned Decision Tree val MAE
0.1509 vs. tuned Random Forest val MAE 0.1704 (tuning still doesn't
beat the untuned baseline — expected, matches `docs/PHASE4_TUNING.md`);
test MAE 0.2356 (RF) vs. 0.2358 (DT); Random Forest (tuned) selected
as final model, registered as `wellpulse_final_model` v1, promoted to
Production. **Do not ship this sandbox's regenerated `mlflow.db`/
`mlruns/` back to Fahad** — same reasoning, other direction; he
regenerates locally the same way.

## What was built

```
src/api/db.py                Engine/session, DATABASE_URL env-var-first, init_db()
src/api/db_models.py         User, CheckIn, Prediction, PredictionExplanation (SQLAlchemy 2.0)
src/api/auth.py              POST /auth/register, X-API-Key auth dependency
src/api/checkins.py          POST /checkins, GET /checkins, GET /checkins/{id}
src/api/main.py               init_db() added to lifespan; both routers mounted
src/api/schemas.py            RegisterRequest/Response, CheckInSummary, CheckInDetail
tests/test_checkins.py        10 new tests, isolated per-test SQLite DB
```

`/api/v1/predict` (Phase 7) is untouched — it stays the free,
unauthenticated, non-persisting "try it" endpoint. `/checkins` is the
new authenticated, persisted path, and it calls
`ModelService.predict_one()` directly rather than reimplementing any
part of feature prep / SHAP / confidence interval / risk tier /
recommendations.

**Schema reuse:** `POST /checkins` takes `PredictRequest` as its body
type directly — no second near-identical schema, for the same reason
Section 18 already gave for the encoder (a hand-rolled translation
layer is where skew creeps in).

**One-to-one modeling:** `CheckIn → Prediction → PredictionExplanation`
is three tables joined by unique foreign keys (`check_in_id`,
`prediction_id`), not one wide table — matches Section 20's own
three-table split, and keeps each table's reason for existing
distinct (raw input vs. model output vs. explanation).

**Recommendations, persisted too:** `prediction_explanations` also
stores `recommendations`, not just `top_factors`/`explanation_sentence`.
Same justification Section 20 gives for storing SHAP output at all
("show the same explanation again later without recomputing it") —
Section 15's recommendation lookup was already flagged as a Phase 7
addition; storing its output is that same addition, extended for the
same stated reason.

**404, not 403, for another user's check-in:** `GET /checkins/{id}`
returns 404 whether the ID doesn't exist or belongs to someone else —
doesn't confirm another user's check-in ID is real.

## Verification

- Full test suite: **59/59 passed** (49 existing + 10 new).
- Live `uvicorn` process, real HTTP requests, in order:
  - `GET /health` → real Production model loaded.
  - `POST /checkins` with no key → `401`.
  - `POST /auth/register` → real per-user API key.
  - Duplicate email → `409`.
  - `POST /checkins` with the key → real prediction (score 5.944,
    medium_risk, 3 SHAP factors, 1 recommendation), persisted, `id: 1`
    returned.
  - `GET /checkins/1` → byte-identical to the submission response.
  - `GET /checkins` → one-row history for that user.
  - Second user registered → their `GET /checkins` is `[]`.
  - Second user requesting the first user's `/checkins/1` → `404`.
- `tests/test_checkins.py` isolates each test's DB by monkeypatching
  `src.api.db`'s module-level `engine`/`SessionLocal` onto a fresh
  temp-file SQLite database *before* the app's lifespan runs
  `init_db()` — mirrors `tests/conftest.py`'s existing MLflow
  isolation pattern rather than inventing a new one. Overrides the
  autouse MLflow-isolation fixture with a no-op (same reason
  `test_api.py` does): `/checkins` needs the real registered
  Production model.

## Not built this phase (by design)

- Alembic/migration framework — `init_db()` just creates tables that
  don't exist yet; outside Section 20/34's stated MVP scope.
- `model_versions`, cohort/consent/counselor tables, free-text notes
  — Section 20's own exclusions, still correct here.
- Password hashing, JWT, sessions, RBAC — Section 19 explicitly scopes
  these out of the MVP.
- Postgres — deferred to Phase 10 per the resolved DB-choice decision
  above; `DATABASE_URL` makes it a config change when that phase
  arrives.
