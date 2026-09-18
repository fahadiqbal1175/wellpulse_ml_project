"""
Phase 13 — Monitoring & Drift: CLI entrypoint for `make build-reference`.

Loads the (already Pandera-validated) training data, runs it through
the EXACT same feature-prep + Production model the serving path uses
(src/api/inference.py's ModelService — same encoder, same
feature_columns, same model), and writes
reports/monitoring/reference_distribution.json. This means the
"training-time predictions" in that file are genuinely what the live
model produces on its own training inputs, not a separately
maintained approximation of it — same training/serving-consistency
principle Section 18 already enforces for a single live request,
just applied to the whole training set at once.

Run this once now, and again any time the model is retrained (Phase
14) — a stale reference would compare live traffic against a model
that's no longer in Production.
"""
from __future__ import annotations

import pandas as pd

from src.api.inference import ModelService
from src.data.schema import validate_raw_dataset
from src.features.engineering import add_engineered_features
from src.monitoring.reference import (
    RAW_FEATURE_COLUMNS,
    build_reference_distribution,
    save_reference_distribution,
)

DATA_PATH = "data/raw/students_social_media_addiction.csv"


def main() -> None:
    raw = validate_raw_dataset(pd.read_csv(DATA_PATH))

    service = ModelService()
    service.load()
    if not service.is_ready:
        raise RuntimeError(f"Could not load the Production model: {service.load_error}")

    feature_df = raw[RAW_FEATURE_COLUMNS]
    engineered = add_engineered_features(feature_df)
    encoded = service.encoder.transform(engineered)
    X = encoded[service.feature_columns]
    predictions = service.model.predict(X)

    reference = build_reference_distribution(feature_df, predictions)
    save_reference_distribution(reference)
    print(
        f"Wrote reference distribution from {len(feature_df)} training rows "
        f"(model {service.model_name} {service.model_version}) to "
        f"reports/monitoring/reference_distribution.json"
    )


if __name__ == "__main__":
    main()