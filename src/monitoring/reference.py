"""
Phase 13 — Monitoring & Drift (Sections 25/26): builds and loads the
training-time reference distribution that live check-in traffic is
compared against.

Run once now, and again any time the model is retrained (Phase 14):
    make build-reference
which writes reports/monitoring/reference_distribution.json. That
file is committed to git (same reasoning as models/final_model.joblib
— see Phase 10's Dockerfile comment) so the scheduled drift job
(run_drift_check.py) never needs to recompute it or load the model
itself — it only ever reads this file + the DB.

N_BINS=5, not the more obvious 10: tested against this project's
actual minimum window size (30 samples) and 10 bins gave roughly a
40% chance of at least one bin landing empty by pure sampling luck
(Poisson(3) per bin), which the PSI formula's epsilon-floor punishes
heavily — a false "significant" flag on ordinary traffic. 5 bins
drops that to ~1%, while a genuinely shifted batch still produces a
PSI an order of magnitude past the 0.25 threshold. Verified both
cases numerically before settling on this.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PROJECT_ROOT / "reports" / "monitoring" / "reference_distribution.json"

# Same 9 fields PredictRequest exposes (src/api/schemas.py) — the raw
# columns the model actually consumes, excluding Student_ID /
# Affects_Academic_Performance / Mental_Health_Score / Addicted_Score
# (ID/leakage/target columns per LEAKAGE_AND_ID_COLS).
NUMERIC_FEATURES = [
    "Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Conflicts_Over_Social_Media",
]
CATEGORICAL_FEATURES = [
    "Gender", "Academic_Level", "Country", "Most_Used_Platform", "Relationship_Status",
]
RAW_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

N_BINS = 5


def _quantile_bin_edges(values: pd.Series, n_bins: int = N_BINS) -> list[float]:
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = values.quantile(quantiles).unique().tolist()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _numeric_reference(values: pd.Series) -> dict:
    edges = _quantile_bin_edges(values)
    counts = pd.cut(values, bins=edges, include_lowest=True).value_counts(sort=False)
    proportions = (counts / counts.sum()).tolist()
    return {"bin_edges": edges, "reference_proportions": proportions}


def _categorical_reference(values: pd.Series) -> dict:
    known = sorted(values.unique().tolist())
    props = values.value_counts(normalize=True).to_dict()
    reference_proportions = {cat: props.get(cat, 0.0) for cat in known}
    reference_proportions["__unexpected__"] = 0.0  # never seen in training, by definition
    return {"known_categories": known, "reference_proportions": reference_proportions}


def build_reference_distribution(training_df: pd.DataFrame, training_predictions: np.ndarray) -> dict:
    numeric = {f: _numeric_reference(training_df[f]) for f in NUMERIC_FEATURES}
    categorical = {f: _categorical_reference(training_df[f]) for f in CATEGORICAL_FEATURES}

    pred_series = pd.Series(training_predictions)
    prediction = _numeric_reference(pred_series)
    prediction["reference_mean"] = float(pred_series.mean())
    prediction["reference_std"] = float(pred_series.std())

    return {
        "built_from_n_rows": len(training_df),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "prediction": prediction,
    }


def save_reference_distribution(reference: dict) -> None:
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Python's json module writes -inf/+inf as Infinity/-Infinity and
    # reads them straight back the same way — not strict JSON, fine
    # for a file this codebase only ever reads with json.loads itself.
    REFERENCE_PATH.write_text(json.dumps(reference, indent=2))


def load_reference_distribution() -> dict:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"{REFERENCE_PATH} not found — run `make build-reference` first.")
    return json.loads(REFERENCE_PATH.read_text())