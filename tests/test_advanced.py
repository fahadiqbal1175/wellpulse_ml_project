"""Phase 4 verification (part 1): LightGBM/XGBoost are added to the
comparison through the same training/evaluation loop as Phase 3, and
the resulting leaderboard still behaves sensibly."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.schema import validate_raw_dataset
from src.data.split import stratified_split
from src.models.advanced import ALL_MODEL_FACTORIES, run_phase4_leaderboard
from src.models.baseline import MODEL_FACTORIES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "students_social_media_addiction.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    if not RAW_CSV.exists():
        pytest.skip(f"{RAW_CSV} not found — run `python -m src.data.ingest` first.")
    return validate_raw_dataset(pd.read_csv(RAW_CSV))


def test_advanced_models_added_to_phase3_set() -> None:
    assert set(ALL_MODEL_FACTORIES) == set(MODEL_FACTORIES) | {"LightGBM", "XGBoost"}
    assert len(ALL_MODEL_FACTORIES) == len(MODEL_FACTORIES) + 2


def test_phase4_leaderboard_has_nine_models(raw_df: pd.DataFrame) -> None:
    split = stratified_split(raw_df)
    leaderboard = run_phase4_leaderboard(split.train, split.val)
    assert set(leaderboard["model"]) == set(ALL_MODEL_FACTORIES)
    assert {"MAE", "RMSE", "R2"}.issubset(leaderboard.columns)


def test_dummy_baselines_still_beaten_by_every_real_model(raw_df: pd.DataFrame) -> None:
    split = stratified_split(raw_df)
    leaderboard = run_phase4_leaderboard(split.train, split.val).set_index("model")

    dummy_mae = min(
        leaderboard.loc["Dummy (mean)", "MAE"], leaderboard.loc["Dummy (median)", "MAE"]
    )
    real_models = [m for m in ALL_MODEL_FACTORIES if not m.startswith("Dummy")]
    for model_name in real_models:
        assert leaderboard.loc[model_name, "MAE"] < dummy_mae
