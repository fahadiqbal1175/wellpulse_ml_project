.PHONY: data test eda baseline advanced tune evaluate error-analysis explain phase5

# Section 7: "a `make data` script pulls/copies the raw file into
# data/raw/, deterministically, with no manual steps" (beyond the one
# unavoidable manual step of downloading the CSV from Kaggle once,
# since Kaggle requires authentication).
#
# Usage:
#   make data                              # re-stamp/re-validate data/raw/students_social_media_addiction.csv
#   make data SOURCE=~/Downloads/students.csv   # copy + validate + stamp from a fresh download
data:
	python3 -m src.data.ingest $(if $(SOURCE),--source $(SOURCE),)

test:
	python3 -m pytest tests/ -v

# Section 37 Phase 2: rebuilds notebooks/01_eda.ipynb from
# scripts/build_eda_notebook.py and executes it top-to-bottom, so the
# committed notebook always has fresh, reproduced outputs.
eda:
	python3 scripts/build_eda_notebook.py

# Section 37 Phase 3: trains 7 model families on the stratified
# train/val split and writes the leaderboard + country-generalization
# check to reports/.
baseline:
	python3 -m src.models.baseline

# Phase 4, step 1: extends the leaderboard to 9 model families
# (adds LightGBM/XGBoost), re-run through the same train/val loop as
# `make baseline`. Writes reports/phase4_leaderboard.csv.
advanced:
	python3 -m src.models.advanced

# Phase 4, step 2: randomized search on the top-2 leaderboard
# candidates (picked fresh from reports/phase4_leaderboard.csv, not
# hardcoded). Writes reports/phase4_tuning_summary.csv,
# reports/experiments/phase4_tuning.csv, and models/phase4_best_model.joblib.
# Requires `make advanced` to have been run first.
tune:
	python3 -m src.models.tuning

# Phase 5, step 1: the FIRST and ONLY time the test fold is used —
# evaluates both the untuned Decision Tree and the tuned Random Forest
# against test, picks the lower-MAE candidate as the final model, and
# saves models/final_model.joblib (model + encoder + feature_columns).
# Requires `make tune` to have been run first.
evaluate:
	python3 -m src.evaluation.final_evaluation

# Phase 5, step 2: residual plots, subgroup performance (country,
# academic level), and the 10 worst-predicted test rows, all against
# the exact same final-model predictions `make evaluate` produced.
error-analysis:
	python3 -m src.evaluation.error_analysis

# Phase 5, step 3: SHAP summary plot + 3 sample explanations (one per
# risk tier), each with a confidence-interval proxy and a templated
# explanation sentence.
explain:
	python3 -m src.evaluation.explainability

# Runs all three Phase 5 steps in order.
phase5: evaluate error-analysis explain
