"""
Data contract for the raw WellPulse dataset.

This schema is the Phase 1 "data contract" artifact from the blueprint
(Section 8 / Phase 1): it exists to catch structural and value-range
problems in the raw Kaggle CSV *before* they reach EDA or model
training, not to encode every nuance discovered later during EDA.

Source dataset: "Students' Social Media Addiction" (Kaggle,
adilshamim8/social-media-addiction-vs-relationships), per Section 4
of the blueprint.

Range/category bounds below come from the real values observed in
data/raw/students_social_media_addiction.csv (e.g. Age 18-24,
Avg_Daily_Usage_Hours 1.5-8.5), padded to reasonable bounds so a
legitimate new respondent isn't rejected, while still catching
impossible values (negative hours, a 400-year-old student, an
unrecognized category, a null in a required field, a duplicate ID).

Anything *plausible-but-unusual* (e.g. an 8-hour sleeper who reports
16 usage hours/day) is deliberately left to the EDA outlier-analysis
step (Phase 2 / Section 8), which flags rather than silently drops.
This schema only rejects what should never occur in a valid raw row.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

# Categories actually observed in the primary dataset. "Other" is
# included for Gender since it's a standard demographic option that
# simply doesn't happen to appear in this particular sample.
KNOWN_GENDERS = ["Male", "Female", "Other"]
KNOWN_ACADEMIC_LEVELS = ["High School", "Undergraduate", "Graduate"]
KNOWN_PLATFORMS = [
    "Instagram", "Twitter", "TikTok", "YouTube", "Facebook",
    "LinkedIn", "Snapchat", "LINE", "KakaoTalk", "VKontakte",
    "WhatsApp", "WeChat",
]
KNOWN_YES_NO = ["Yes", "No"]
KNOWN_RELATIONSHIP_STATUSES = ["Single", "In Relationship", "Complicated"]


class RawStudentRecordSchema(pa.DataFrameModel):
    """One row = one survey respondent (a single cross-sectional snapshot)."""

    Student_ID: Series[int] = pa.Field(ge=1, unique=True)
    Age: Series[int] = pa.Field(ge=10, le=100)
    Gender: Series[str] = pa.Field(isin=KNOWN_GENDERS)
    Academic_Level: Series[str] = pa.Field(isin=KNOWN_ACADEMIC_LEVELS)
    Country: Series[str] = pa.Field(str_length={"min_value": 2, "max_value": 60})
    Avg_Daily_Usage_Hours: Series[float] = pa.Field(ge=0, le=24)
    Most_Used_Platform: Series[str] = pa.Field(isin=KNOWN_PLATFORMS)
    Affects_Academic_Performance: Series[str] = pa.Field(isin=KNOWN_YES_NO)
    Sleep_Hours_Per_Night: Series[float] = pa.Field(ge=0, le=24)
    Mental_Health_Score: Series[int] = pa.Field(ge=1, le=10)
    Relationship_Status: Series[str] = pa.Field(isin=KNOWN_RELATIONSHIP_STATUSES)
    Conflicts_Over_Social_Media: Series[int] = pa.Field(ge=0, le=20)
    Addicted_Score: Series[int] = pa.Field(ge=1, le=10)

    class Config:
        # No unlisted/extra columns allowed — a schema drift (renamed
        # or added column) should fail loudly, not pass silently.
        strict = True
        # Coerce dtypes (e.g. "6" -> 6) so a formatting quirk doesn't
        # masquerade as a dtype failure; genuinely bad values still
        # fail on the range/isin checks below.
        coerce = True


def validate_raw_dataset(df):
    """
    Validate a raw dataframe against the WellPulse data contract.

    Uses lazy validation so ALL schema violations are collected and
    reported together (not just the first one found) — important for
    diagnosing a corrupted file in one pass instead of a fix-rerun loop.

    Raises:
        pandera.errors.SchemaErrors: if validation fails. The exception's
            `.failure_cases` attribute is a dataframe listing every
            violation (column, check, failing value, row index).

    Returns:
        The validated dataframe (with coerced dtypes) on success.
    """
    return RawStudentRecordSchema.validate(df, lazy=True)
