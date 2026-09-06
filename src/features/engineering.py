"""
Deterministic, per-row feature engineering (Section 9, Phase 2).

These transforms depend only on a single row's raw values — never on
the target, and never on statistics computed across other rows (no
means, no category frequencies). That means they're safe to apply
identically to train/val/test and even to a single live prediction
request, with zero leakage risk. Anything that *does* need fitting on
training data only (categorical encoding, rare-category bucketing)
lives in `src/features/encoding.py` instead — kept deliberately
separate so it's obvious which functions are safe to call anywhere
and which must only ever be fit on the training fold (Section 10).
"""
from __future__ import annotations

import pandas as pd

# Bin edges chosen from the real observed Age range (18-24) in the
# primary dataset, with headroom on both sides for future respondents.
AGE_BIN_EDGES = [0, 17, 20, 23, 200]
AGE_BIN_LABELS = ["under_18", "18_20", "21_23", "24_plus"]


def add_usage_to_sleep_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    usage_to_sleep_ratio = Avg_Daily_Usage_Hours / Sleep_Hours_Per_Night

    A literature-supported combined signal (Section 9): a student with
    high usage AND low sleep is a materially different case than one
    with high usage but plenty of sleep, and the ratio captures that
    in a single feature rather than relying on the model to learn the
    interaction from the two raw columns alone.
    """
    out = df.copy()
    out["usage_to_sleep_ratio"] = (
        out["Avg_Daily_Usage_Hours"] / out["Sleep_Hours_Per_Night"]
    )
    return out


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bins Age into 4 groups. Categorical, so it's one-hot encoded downstream."""
    out = df.copy()
    out["age_group"] = pd.cut(
        out["Age"], bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS, right=True
    ).astype(str)
    return out


def add_usage_conflict_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    usage_conflict_interaction = Avg_Daily_Usage_Hours * Conflicts_Over_Social_Media

    Usage combined with social friction (Section 9) — a student who
    uses social media heavily AND has frequent conflicts over it is a
    combination the two raw features alone don't directly express.
    """
    out = df.copy()
    out["usage_conflict_interaction"] = (
        out["Avg_Daily_Usage_Hours"] * out["Conflicts_Over_Social_Media"]
    )
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all three per-row engineered features in sequence."""
    out = df.pipe(add_usage_to_sleep_ratio).pipe(add_age_group).pipe(
        add_usage_conflict_interaction
    )
    return out


# Thresholds derived empirically in the Phase 2 EDA notebook
# (notebooks/01_eda.ipynb, Section 5): the 25th/75th percentiles of
# the observed Mental_Health_Score distribution (4-9 range). An
# initial score<=4 / 5-7 / >=8 split put only 4.1% of rows in
# high-risk (score=4 is the observed minimum); this quantile-based
# split gives 28.7% / 56.3% / 15.0% instead — every tier has
# meaningful representation. This is a display-only transform
# (Section 5), never a separately trained classifier, and is the
# single source of truth used both by the EDA notebook and by
# src/data/split.py for stratification.
def compute_risk_tier(df: pd.DataFrame) -> pd.Series:
    """Maps Mental_Health_Score to a low/medium/high risk tier (display-only)."""

    def _to_tier(score: float) -> str:
        if score <= 5:
            return "high_risk"
        elif score <= 7:
            return "medium_risk"
        return "low_risk"

    return df["Mental_Health_Score"].apply(_to_tier)
