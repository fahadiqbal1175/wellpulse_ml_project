"""
Data splitting (Section 10, used going into Phase 3).

Two independent splits, built for two different purposes — never
mixed with each other:

1. `stratified_split()` — the PRIMARY train/val/test split (70/15/15),
   stratified on the risk tier so all three splits have a similar
   score distribution. This is what Phase 3's baseline leaderboard and
   every later model-comparison step uses. The test fold from this
   split is intentionally NOT touched again until Phase 5's final
   evaluation — comparing models against `val`, not `test`, avoids
   quietly "peeking" at test performance across dozens of baseline
   and tuning runs, which would make the final test-set number an
   overly optimistic estimate of real generalization.

2. `country_generalization_split()` — a SEPARATE, self-contained
   diagnostic split that holds out entire countries (not just rows),
   used only to answer "does the model generalize beyond the sampled
   countries?" (Section 10's explicit ask, motivated by the bias risk
   noted in Section 4: self-selected survey respondents concentrated
   in specific countries). It is never used for model selection or
   leaderboard ranking — only reported alongside it as a supplementary
   check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.engineering import compute_risk_tier

RANDOM_STATE = 42


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> SplitResult:
    """
    70/15/15 train/val/test split (defaults), stratified on the risk
    tier (Section 10) so every split has a comparable score
    distribution. Two-step split under the hood: carve off `test`
    first, then carve `val` off what remains.
    """
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    tier = compute_risk_tier(df)

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=tier, random_state=random_state
    )
    # val_size was defined as a fraction of the FULL dataset; convert
    # it to a fraction of what remains after removing the test split.
    remaining_val_fraction = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=remaining_val_fraction,
        stratify=compute_risk_tier(train_val_df),
        random_state=random_state,
    )
    return SplitResult(train=train_df, val=val_df, test=test_df)


def country_generalization_split(
    df: pd.DataFrame,
    holdout_fraction: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Holds out entire countries (not just rows) to test whether a model
    generalizes beyond the countries it was trained on. Returns
    (train_countries_df, holdout_countries_df) with zero country
    overlap between the two by construction.

    This is intentionally separate from `stratified_split()` above —
    it is a supplementary diagnostic, not part of the primary
    train/val/test used for model selection.
    """
    rng = np.random.RandomState(random_state)
    countries = np.array(df["Country"].unique(), dtype=object)
    rng.shuffle(countries)

    n_holdout = max(1, int(round(len(countries) * holdout_fraction)))
    holdout_countries = set(countries[:n_holdout])

    holdout_df = df[df["Country"].isin(holdout_countries)]
    train_df = df[~df["Country"].isin(holdout_countries)]
    return train_df, holdout_df
