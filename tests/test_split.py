"""Phase 3 verification: the primary split has no row overlap and
preserves the risk-tier distribution across folds; the country
generalization split has zero country overlap by construction."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.schema import validate_raw_dataset
from src.data.split import country_generalization_split, stratified_split
from src.features.engineering import compute_risk_tier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "students_social_media_addiction.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    if not RAW_CSV.exists():
        pytest.skip(f"{RAW_CSV} not found — run `python -m src.data.ingest` first.")
    return validate_raw_dataset(pd.read_csv(RAW_CSV))


def test_stratified_split_has_no_row_overlap(raw_df: pd.DataFrame) -> None:
    split = stratified_split(raw_df)
    ids_train = set(split.train["Student_ID"])
    ids_val = set(split.val["Student_ID"])
    ids_test = set(split.test["Student_ID"])

    assert not (ids_train & ids_val)
    assert not (ids_train & ids_test)
    assert not (ids_val & ids_test)
    assert len(ids_train) + len(ids_val) + len(ids_test) == len(raw_df)


def test_stratified_split_preserves_tier_distribution(raw_df: pd.DataFrame) -> None:
    split = stratified_split(raw_df)
    full_dist = compute_risk_tier(raw_df).value_counts(normalize=True)

    for part in (split.train, split.val, split.test):
        part_dist = compute_risk_tier(part).value_counts(normalize=True)
        # Every tier's proportion should be within 5 percentage points
        # of the full dataset's — a loose bound that still catches a
        # broken/non-stratified split.
        for tier in full_dist.index:
            assert abs(part_dist.get(tier, 0) - full_dist[tier]) < 0.05


def test_stratified_split_rejects_invalid_sizes(raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        stratified_split(raw_df, val_size=0.6, test_size=0.6)


def test_country_split_has_zero_country_overlap(raw_df: pd.DataFrame) -> None:
    train_df, holdout_df = country_generalization_split(raw_df)
    train_countries = set(train_df["Country"])
    holdout_countries = set(holdout_df["Country"])

    assert not (train_countries & holdout_countries)
    assert len(train_df) + len(holdout_df) == len(raw_df)
    assert len(holdout_countries) > 0
