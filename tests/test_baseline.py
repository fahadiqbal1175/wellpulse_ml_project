"""Phase 3 verification: at least 4 model families are trained and
compared on the same split (Milestone ML-3), and the leaderboard
behaves sensibly (dummy baselines are the worst performers)."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.schema import validate_raw_dataset
from src.data.split import stratified_split
from src.models.baseline import (
    MODEL_FACTORIES,
    build_model_ready_xy,
    run_baseline_leaderboard,
    run_country_generalization_check,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "students_social_media_addiction.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    if not RAW_CSV.exists():
        pytest.skip(f"{RAW_CSV} not found — run `python -m src.data.ingest` first.")
    return validate_raw_dataset(pd.read_csv(RAW_CSV))


def test_at_least_four_model_families_registered() -> None:
    # Milestone ML-3: at least 4 distinct model families compared on the same split.
    assert len(MODEL_FACTORIES) >= 4


def test_build_model_ready_xy_drops_leakage_columns(raw_df: pd.DataFrame) -> None:
    X, y, encoder = build_model_ready_xy(raw_df, encoder=None, fit=True)
    assert "Addicted_Score" not in X.columns
    assert "Affects_Academic_Performance" not in X.columns
    assert "Student_ID" not in X.columns
    assert "Mental_Health_Score" not in X.columns
    assert len(y) == len(X) == len(raw_df)


def test_val_encoding_reuses_train_fitted_encoder_with_matching_columns(
    raw_df: pd.DataFrame,
) -> None:
    split = stratified_split(raw_df)
    X_train, _, encoder = build_model_ready_xy(split.train, encoder=None, fit=True)
    X_val, _, _ = build_model_ready_xy(split.val, encoder=encoder, fit=False)
    assert list(X_train.columns) == list(X_val.columns)


def test_leaderboard_has_all_registered_models(raw_df: pd.DataFrame) -> None:
    split = stratified_split(raw_df)
    leaderboard = run_baseline_leaderboard(split.train, split.val)
    assert set(leaderboard["model"]) == set(MODEL_FACTORIES.keys())
    assert {"MAE", "RMSE", "R2"}.issubset(leaderboard.columns)


def test_dummy_baselines_are_beaten_by_every_real_model(raw_df: pd.DataFrame) -> None:
    split = stratified_split(raw_df)
    leaderboard = run_baseline_leaderboard(split.train, split.val).set_index("model")

    dummy_mae = min(
        leaderboard.loc["Dummy (mean)", "MAE"], leaderboard.loc["Dummy (median)", "MAE"]
    )
    real_models = [m for m in MODEL_FACTORIES if not m.startswith("Dummy")]
    for model_name in real_models:
        assert leaderboard.loc[model_name, "MAE"] < dummy_mae


def test_country_generalization_check_returns_two_rows(raw_df: pd.DataFrame) -> None:
    result = run_country_generalization_check(raw_df)
    assert len(result) == 2
    assert set(result["eval_set"]) == {
        "seen_countries (held-back rows)",
        "unseen_countries (fully held out)",
    }
