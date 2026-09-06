"""Phase 2 verification: engineered features are correct, and the
fit-on-train-only encoder generalizes to unseen categories without
leaking train statistics into val/test."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.schema import validate_raw_dataset
from src.features.encoding import CategoricalFeatureEncoder, RareCategoryBucketer
from src.features.engineering import add_engineered_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "students_social_media_addiction.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    if not RAW_CSV.exists():
        pytest.skip(f"{RAW_CSV} not found — run `python -m src.data.ingest` first.")
    return validate_raw_dataset(pd.read_csv(RAW_CSV))


def test_usage_to_sleep_ratio_is_correct(raw_df: pd.DataFrame) -> None:
    out = add_engineered_features(raw_df)
    expected = raw_df["Avg_Daily_Usage_Hours"] / raw_df["Sleep_Hours_Per_Night"]
    assert (out["usage_to_sleep_ratio"] - expected).abs().max() < 1e-9


def test_usage_conflict_interaction_is_correct(raw_df: pd.DataFrame) -> None:
    out = add_engineered_features(raw_df)
    expected = raw_df["Avg_Daily_Usage_Hours"] * raw_df["Conflicts_Over_Social_Media"]
    assert (out["usage_conflict_interaction"] - expected).abs().max() < 1e-9


def test_age_group_covers_every_row_with_no_nulls(raw_df: pd.DataFrame) -> None:
    out = add_engineered_features(raw_df)
    assert out["age_group"].isnull().sum() == 0
    assert set(out["age_group"].unique()).issubset(
        {"under_18", "18_20", "21_23", "24_plus"}
    )


def test_rare_category_bucketer_maps_unseen_values_to_other() -> None:
    train = pd.Series(["USA", "USA", "USA", "India", "India", "UK"])
    bucketer = RareCategoryBucketer(top_n=2).fit(train)
    # USA and India are the top-2 seen in training; UK and a never-seen
    # country should both fall back to "Other".
    result = bucketer.transform(pd.Series(["USA", "India", "UK", "Nowhereland"]))
    assert result.tolist() == ["USA", "India", "Other", "Other"]


def test_categorical_encoder_fit_on_train_transforms_val_consistently(
    raw_df: pd.DataFrame,
) -> None:
    """
    The encoder must be fit ONLY on a train split and then reused on
    val/test — this test simulates exactly that split discipline
    (Section 10) and checks val doesn't silently gain/lose columns
    just because it contains categories absent from train.
    """
    engineered = add_engineered_features(raw_df).drop(
        columns=["Addicted_Score", "Affects_Academic_Performance", "Student_ID"]
    )
    train = engineered.iloc[:500]
    val = engineered.iloc[500:]

    encoder = CategoricalFeatureEncoder()
    train_encoded = encoder.fit(train).transform(train)
    val_encoded = encoder.transform(val)

    # Same columns, same order, regardless of what categories val contains.
    assert list(train_encoded.columns) == list(val_encoded.columns)
    # No row should be lost or duplicated in either split.
    assert len(train_encoded) == len(train)
    assert len(val_encoded) == len(val)


def test_encoder_raises_if_transform_called_before_fit() -> None:
    bucketer = RareCategoryBucketer()
    with pytest.raises(RuntimeError):
        bucketer.transform(pd.Series(["USA"]))
