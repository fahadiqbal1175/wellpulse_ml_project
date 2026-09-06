"""Phase 4 verification (part 2): candidate selection is computed from
the leaderboard (not hardcoded), randomized search runs end-to-end on
real data, and every trial gets logged."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.schema import validate_raw_dataset
from src.data.split import stratified_split
from src.experiments.logger import ExperimentLogger
from src.models import tuning
from src.models.baseline import build_model_ready_xy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "students_social_media_addiction.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    if not RAW_CSV.exists():
        pytest.skip(f"{RAW_CSV} not found — run `python -m src.data.ingest` first.")
    return validate_raw_dataset(pd.read_csv(RAW_CSV))


def test_select_tuning_candidates_picks_top_two_with_param_space() -> None:
    leaderboard = pd.DataFrame(
        {
            "model": ["Dummy (mean)", "LightGBM", "Decision Tree", "Random Forest"],
            "MAE": [0.9, 0.14, 0.15, 0.19],
        }
    )
    # Dummy has no param space and must never be selected, even with a
    # (hypothetically) lower MAE than everything else.
    candidates = tuning.select_tuning_candidates(leaderboard, top_n=2)
    assert candidates == ["LightGBM", "Decision Tree"]


def test_tune_model_end_to_end_logs_every_trial(raw_df: pd.DataFrame, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tuning, "N_ITER", 4)
    monkeypatch.setattr(tuning, "CV_FOLDS", 2)

    split = stratified_split(raw_df)
    X_train, y_train, encoder = build_model_ready_xy(split.train, encoder=None, fit=True)
    X_val, y_val, _ = build_model_ready_xy(split.val, encoder=encoder, fit=False)

    logger = ExperimentLogger(experiment="test_phase4_tuning", output_dir=tmp_path)
    result = tuning.tune_model("Decision Tree", X_train, y_train, X_val, y_val, logger)

    assert "best_params" in result
    assert "MAE" in result

    logged = pd.read_csv(logger.csv_path)
    # 4 CV trials + 1 best-of-search summary row
    assert len(logged) == 4 + 1
