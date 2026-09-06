# WellPulse v2 — ML-First

Predicts a continuous, self-reported student wellbeing/mental-health
score from behavioral/demographic survey data (social-media usage,
sleep, platform, academic/relationship context), served through a
tracked, registered, monitored ML pipeline. See
`wellpulse-ml-blueprint-v2.md` for the full specification.

## Status

**Phase 0 — Problem & Dataset Validation:** done (confirmed below).
**Phase 1 — Data Pipeline:** done.
**Phase 2 — EDA & Feature Engineering:** done.
**Phase 3 — Baseline Models:** done.
**Phase 4 — Advanced Modeling & Tuning:** done.

## Phase 0/1 notes

Primary dataset: **Students' Social Media Addiction** (Kaggle,
[`adilshamim8/social-media-addiction-vs-relationships`](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships)).
Confirmed locally: **705 rows, 13 columns, zero missing values, zero
duplicate `Student_ID`s** — matches the blueprint's Section 4 claims.
One thing the blueprint didn't call out: the 705 rows span **110
distinct countries** (~6.4 rows/country on average) — worth keeping in
mind for the Section 10 grouped-by-country generalization split in
Phase 2/3, since most countries will have too few rows to split on
individually.

Kaggle requires authentication to download directly, so `data/raw/students_social_media_addiction.csv`
here was sourced from a GitHub mirror of the exact same dataset
([`rizanpradiya/Analyzing-Social-Media-Addiction-Among-Students`](https://github.com/rizanpradiya/Analyzing-Social-Media-Addiction-Among-Students))
with matching column names and row count. If you'd rather pull it
straight from Kaggle yourself (e.g. via `kaggle datasets download`),
just point the ingestion script at that file with `--source`.

## Setup

```bash
pip install -r requirements.txt
```

## Data pipeline (Phase 1)

```bash
# Re-validate/re-stamp the file already at data/raw/:
make data

# Or ingest a fresh download from wherever it landed:
make data SOURCE=~/Downloads/students.csv

# Run the test suite (includes the corrupted-copy verification):
make test
```

`src/data/ingest.py` copies the raw CSV to a canonical path under
`data/raw/`, validates it against the data contract in
`src/data/schema.py`, and writes `data/raw/dataset_version.txt` — a
content hash + ingestion month (e.g. `396c5612_2026-09`) that every
later MLflow run will log, so any experiment can be traced back to the
exact CSV that produced it (Section 7).

`src/data/schema.py` is a Pandera `DataFrameModel` data contract:
column types, plausible value ranges, and known categories, with
`strict=True` so an added/renamed/dropped column fails loudly instead
of silently changing what downstream code sees. It intentionally does
**not** encode outlier judgment calls (e.g. an unusually high but
possible usage-hours value) — that belongs to Phase 2's EDA, which
flags rather than silently drops.

`tests/test_data_validation.py` verifies both directions: the real
dataset passes as-is, and several distinct deliberately-introduced
corruptions (negative age, out-of-range score, unrecognized category,
null in a required field, duplicate ID, extra column, missing column)
are all caught in one lazy-validation pass.

## EDA & feature engineering (Phase 2)

```bash
# One-time setup, only needed if you want to REBUILD the notebook:
python3 -m ipykernel install --user --name python3 --display-name "Python 3"

# Rebuild + re-execute the EDA notebook top to bottom:
make eda
```

`notebooks/01_eda.ipynb` is generated (not hand-edited) from
`scripts/build_eda_notebook.py`, then executed end-to-end so its
outputs are always real and reproducible, not stale. It covers
missingness, univariate/bivariate distributions, the correlation
matrix, outlier scanning, risk-tier balance, and — most importantly —
**the leakage check**.

**Key finding:** `Addicted_Score` was found to explain 89.3% of the
target's variance *on its own* (a near-tautological relationship) and
is **excluded** from the model's feature set as a result.
`Affects_Academic_Performance` (already outside the blueprint's
Section 5 input list) was independently confirmed as correctly
excluded (65.4% variance alone). Full reasoning, numbers, and the
finalized feature table are in `docs/FEATURES.md`.

Also worth knowing: even without `Addicted_Score`, the remaining
behavioral features explain ~82% of the target's variance with a
plain linear fit — unusually clean for organic self-report survey
data. This is documented as a dataset-quality caveat, not hidden (see
`docs/FEATURES.md`).

`src/features/engineering.py` — deterministic, per-row transforms
(`usage_to_sleep_ratio`, `age_group`, `usage_conflict_interaction`),
safe to apply to any split or a single live request.

`src/features/encoding.py` — fit-on-train-only categorical encoding
(`RareCategoryBucketer` + `CategoricalFeatureEncoder`). Not fit yet —
that happens in Phase 3 once there's an actual train split — but built
now so the feature list is finalized. Buckets `Country` (110 raw
values for 705 rows) into the top 8 + "Other" before one-hot encoding,
to avoid dozens of near-empty single-respondent columns.

`tests/test_feature_engineering.py` — verifies the engineered features
are computed correctly and that the encoder generalizes to categories
unseen in "train" without leaking train-only information into "val".

## Baseline models (Phase 3)

```bash
make baseline
```

Splits the data 70/15/15 (stratified on risk tier, `src/data/split.py`)
and trains 7 model families — dummy mean/median, Linear/Ridge/Lasso,
Decision Tree, Random Forest — comparing all of them on the **val**
set only (test stays untouched until Phase 5's final evaluation).
Writes `reports/phase3_leaderboard.csv` and
`reports/phase3_country_generalization.csv`.

**Current leader (by MAE, the blueprint's primary metric): Decision
Tree** (MAE 0.151) — though Random Forest has better RMSE/R², a
genuine tension documented in `docs/BASELINE_RESULTS.md` rather than
picked around.

**Country generalization check:** a Random Forest's error roughly
**doubles** (MAE 0.16 → 0.39) on countries it never saw during
training — a real, measured confirmation of the sampling-bias risk
flagged back in Section 4. Full numbers and discussion in
`docs/BASELINE_RESULTS.md`.

`src/data/split.py` — the stratified train/val/test split, plus the
separate country-holdout split used only for the generalization check.

`src/models/baseline.py` — trains and evaluates all 7 model families,
reusing the Phase 2 feature engineering/encoding modules with the
encoder fit on train only.

`tests/test_split.py`, `tests/test_baseline.py` — verify zero
row/country overlap across splits, stratification is preserved, all
dummy baselines are beaten by every real model, and at least 4 model
families are compared (Milestone ML-3).

## Advanced modeling & tuning (Phase 4)

```bash
make advanced   # extend the leaderboard to 9 model families (adds LightGBM/XGBoost)
make tune       # randomized search on the top-2 candidates by val MAE
```

`make advanced` re-runs Phase 3's exact training/evaluation loop
(`src.models.baseline.run_baseline_leaderboard`, now given an optional
`model_factories` argument) over 9 model families instead of 7,
writing `reports/phase4_leaderboard.csv`. **Neither LightGBM nor
XGBoost beat Decision Tree or Random Forest** — unsurprising with only
493 training rows and a dataset already flagged in Phase 2 as unusually
clean for self-report data.

`make tune` runs `RandomizedSearchCV` (60 iterations, 5-fold CV on
train only) over the top-2 candidates by MAE — computed fresh from the
leaderboard each run, not hardcoded, though it confirmed Decision
Tree/Random Forest are still the pair. Every trial is logged via
`src/experiments/logger.py` to `reports/experiments/phase4_tuning.csv`,
in a format Phase 6's MLflow work is meant to read from / replace.
Writes `reports/phase4_tuning_summary.csv` and
`models/phase4_best_model.joblib`.

**Result: tuning did not beat the untuned baseline** (best tuned MAE
0.1704, Random Forest, vs. untuned Decision Tree's 0.1509) — reported
as-is rather than re-run until it "worked," per this project's practice
of documenting inconvenient results. Full discussion of why —
including the case that the untuned tree's win looks partly like a
favorable-split artifact rather than genuinely superior generalization
— is in `docs/PHASE4_TUNING.md`.

`src/models/advanced.py`, `src/models/param_spaces.py`,
`src/models/tuning.py`, `src/experiments/logger.py` — implementation.
`tests/test_advanced.py`, `tests/test_tuning.py` — verification.

## Next: Phase 5 — Final Evaluation

First and only look at the test fold. Given Phase 4's FAIL verdict,
evaluate both the untuned Decision Tree (Phase 3's leaderboard winner)
and the tuned Random Forest (Phase 4's best CV-validated model)
against test, rather than assuming tuning produced the model to carry
forward.
