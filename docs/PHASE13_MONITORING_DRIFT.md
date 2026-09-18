# Phase 13 — Monitoring & Drift (Sections 25/26)

## What this adds
- `src/monitoring/reference.py` + `build_reference.py`: a one-time
  (re-run after any retrain) script that computes the training-time
  feature/prediction distribution and commits it to
  `reports/monitoring/reference_distribution.json`.
- `src/monitoring/drift.py`: pure PSI-based drift computation, unit
  tested in isolation (`tests/test_drift.py`).
- `src/monitoring/run_drift_check.py`: the scheduled job. Reads the
  last 200 check-ins (env `DRIFT_WINDOW_SIZE`) from whichever
  `DATABASE_URL` is set, compares against the committed reference,
  writes `reports/monitoring/latest_drift_report.json` +
  `drift_history.csv`.
- `.github/workflows/drift_check.yml`: runs the job weekly + on
  manual dispatch, against Render Postgres's External Database URL,
  and commits the updated report back to the repo either way.
- `src/monitoring/seed_synthetic_checkins.py`: Milestone ML-9's
  verify tool — seeds deliberately shifted synthetic check-ins.

## Decisions made
- **Rolling window = last N check-ins**, not last N days (default
  N=200, floor of 30 below which the job reports
  `"insufficient_data"` rather than a fabricated verdict). Real
  traffic on a free-tier deploy will be low and uneven; a count-based
  window always has something to compare, a day-based one might not.
- **Drift metric = PSI only**, for numeric features, categorical
  features, and the predicted score alike — one metric, one set of
  thresholds (0.10 / 0.25), no new dependency (no scipy).
- **N_BINS = 5**, not 10: tested against a 30-sample window and found
  10 bins gave roughly a 40% chance of a spurious "significant" flag
  from pure sampling noise (an empty bin, punished by PSI's
  epsilon-floor). 5 bins drops that to ~1% while still flagging a
  genuinely shifted batch by more than an order of magnitude past the
  threshold.
- **The scheduled job never loads the model.** `predicted_score` is
  already persisted by `/checkins` at submission time, so the job's
  only dependencies are pandas + sqlalchemy + psycopg2-binary —
  independent of mlflow/shap/scikit-learn entirely.
- **A flagged drift surfaces two ways**: the report is committed back
  to the repo regardless of outcome, and the Action itself fails
  (red X) when the verdict is "significant" — moderate-tier drift is
  visible in the run summary and the committed report without
  failing the build.

## Known limitations (documented, not glossed over)
- **Only `/checkins` traffic is monitored.** `/api/v1/predict` is
  free, unauthenticated, and by design never persists anything
  (Phase 7/8) — so this job's "incoming traffic" is check-in traffic
  only, not total request volume.
- **No concept/label drift.** True labels aren't collected in
  production (Section 25) — this job only ever measures feature and
  prediction distribution shift, never whether the model's
  predictions are still accurate.
- **PSI thresholds (0.10/0.25) are the conventional heuristic bands,
  not calibrated against this specific dataset/model** — a
  reasonable industry default, not a statistically derived cutoff.
- **The free Render Postgres expires 30 days after creation** (Phase
  12) — the committed `reports/monitoring/` files are the durable
  record even if the database itself gets recreated.

## How to verify (Milestone ML-9)
1. `make build-reference` locally (needs the Production model loaded
   — commit the resulting `reports/monitoring/reference_distribution.json`).
2. Point `DATABASE_URL` at Render's External Database URL and run
   `make seed-drift-test N=50`.
3. `make drift-check` (same `DATABASE_URL`) — expect `verdict:
   "significant"` in `reports/monitoring/latest_drift_report.json`,
   with `Most_Used_Platform` and/or `Sleep_Hours_Per_Night` as the
   worst offenders.
4. Push the reference file, then trigger `.github/workflows/drift_check.yml`
   manually (Actions tab → Run workflow) and confirm it goes red.