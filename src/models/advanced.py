"""
Phase 4 — Advanced Modeling & Tuning (part 1): adds LightGBM and
XGBoost to the Phase 3 comparison and re-produces the leaderboard
through the exact same training/evaluation loop as
`src.models.baseline.run_baseline_leaderboard` (via its new optional
`model_factories` argument), so the resulting 9-model leaderboard is
one single, reproducible run rather than a read-old-CSV-and-append.

Country generalization is intentionally NOT re-run here — Phase 3's
`run_country_generalization_check` already established that pattern;
repeating it for every new model family is later-phase scope.

Usage:
    python -m src.models.advanced
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.data.schema import validate_raw_dataset
from src.data.split import stratified_split
from src.models.baseline import (
    MODEL_FACTORIES,
    RANDOM_STATE,
    RAW_CSV,
    build_model_ready_xy,
    run_baseline_leaderboard,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE4_LEADERBOARD_CSV = PROJECT_ROOT / "reports" / "phase4_leaderboard.csv"

# Same construction pattern as Phase 3's MODEL_FACTORIES: name -> factory,
# no hyperparameter search yet (that's src/models/tuning.py).
ADVANCED_MODEL_FACTORIES = {
    "LightGBM": lambda: LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1),
    "XGBoost": lambda: XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
}

ALL_MODEL_FACTORIES = {**MODEL_FACTORIES, **ADVANCED_MODEL_FACTORIES}


def run_phase4_leaderboard(train_df: pd.DataFrame, val_df: pd.DataFrame, return_models: bool = False):
    return run_baseline_leaderboard(
        train_df, val_df, model_factories=ALL_MODEL_FACTORIES, return_models=return_models
    )


def main() -> None:
    df = validate_raw_dataset(pd.read_csv(RAW_CSV))
    split = stratified_split(df)

    leaderboard, fitted_models = run_phase4_leaderboard(split.train, split.val, return_models=True)
    print("=== Phase 4 Leaderboard (val set, 9 model families) ===")
    print(leaderboard.to_string(index=False))

    PHASE4_LEADERBOARD_CSV.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(PHASE4_LEADERBOARD_CSV, index=False)
    print(f"\nSaved to {PHASE4_LEADERBOARD_CSV}")

    try:
        from src.models.baseline import _log_leaderboard_to_mlflow

        X_train, _, _ = build_model_ready_xy(split.train, encoder=None, fit=True)
        _log_leaderboard_to_mlflow(leaderboard, fitted_models, X_train, phase="phase4_advanced")
        print(f"Logged {len(fitted_models)} model runs to MLflow (experiment: wellpulse_mental_health_score)")
    except Exception as exc:  # pragma: no cover - MLflow logging is best-effort
        print(f"[main] MLflow logging skipped: {exc}")


if __name__ == "__main__":
    main()
