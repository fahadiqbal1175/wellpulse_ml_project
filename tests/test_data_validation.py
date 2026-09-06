"""
Phase 1 verification (Section 37): "validation fails on a deliberately
corrupted copy of the file."

These tests check both directions of the data contract:
  1. The real raw dataset passes validation as-is (a false-positive
     failure here would block every downstream phase).
  2. Several distinct, deliberately-introduced corruptions each cause
     validation to fail, and all of them are reported together in one
     pass (not just the first one found).
"""
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from src.data.schema import validate_raw_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "students_social_media_addiction.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    if not RAW_CSV.exists():
        pytest.skip(
            f"{RAW_CSV} not found — run `python -m src.data.ingest` first."
        )
    return pd.read_csv(RAW_CSV)


def test_real_dataset_passes_validation(raw_df: pd.DataFrame) -> None:
    validated = validate_raw_dataset(raw_df)
    assert len(validated) == len(raw_df)
    assert len(validated.columns) == 13


def test_corrupted_copy_fails_validation(raw_df: pd.DataFrame) -> None:
    corrupted = raw_df.copy()

    # Introduce several distinct, unambiguous violations at once.
    corrupted.loc[0, "Age"] = -5                       # impossible value
    corrupted.loc[1, "Mental_Health_Score"] = 55        # out of the 1-10 scale
    corrupted.loc[2, "Gender"] = "Unknown_Value_X"      # unrecognized category
    corrupted.loc[3, "Sleep_Hours_Per_Night"] = None    # null in a required field
    corrupted.loc[4, "Student_ID"] = corrupted.loc[5, "Student_ID"]  # duplicate ID

    with pytest.raises(pandera.errors.SchemaErrors) as exc_info:
        validate_raw_dataset(corrupted)

    failures = exc_info.value.failure_cases
    failing_columns = set(failures["column"])
    # Every corrupted column should show up as a distinct failure —
    # confirms lazy validation collects all violations, not just the first.
    assert {
        "Student_ID",
        "Age",
        "Gender",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
    }.issubset(failing_columns)


def test_extra_column_is_rejected(raw_df: pd.DataFrame) -> None:
    """strict=True in the schema config should reject schema drift."""
    drifted = raw_df.copy()
    drifted["Unexpected_New_Column"] = "surprise"

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw_dataset(drifted)


def test_missing_column_is_rejected(raw_df: pd.DataFrame) -> None:
    dropped = raw_df.drop(columns=["Addicted_Score"])

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw_dataset(dropped)
