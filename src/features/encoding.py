"""
Fit-on-train-only categorical encoding (Sections 9-10, Phase 2 design
/ Phase 3 usage).

Unlike engineering.py, everything here has a `fit` step that must only
ever see the TRAINING fold (Section 10: "encoding/scaling fit only on
the training fold, never on val/test"). These classes are built now,
during Phase 2, as part of finalizing the feature list — but they are
not fit yet, since there's no train/val/test split until Phase 3.
Phase 3 will do:

    bucketer = RareCategoryBucketer(column="Country", top_n=8).fit(train_df)
    train_df = bucketer.transform(train_df)
    val_df   = bucketer.transform(val_df)
    test_df  = bucketer.transform(test_df)

Why Country needs bucketing (and Platform/Gender/etc. don't): Phase 1's
EDA found 705 rows spread across 110 countries (~6.4 rows/country) —
one-hot encoding all 110 directly would create dozens of near-empty,
effectively single-respondent columns that a model could trivially
(and meaninglessly) memorize. Bucketing rare countries into "Other"
keeps the encoding meaningful and reduces overfitting risk. The other
categoricals (Gender: 2, Academic_Level: 3, Most_Used_Platform: 12,
Relationship_Status: 3) have few enough categories, each with
reasonable support, that they're one-hot encoded directly with no
bucketing needed.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

OTHER_LABEL = "Other"

# Directly one-hot encoded, no bucketing — low cardinality, adequate
# support per category in the primary dataset.
DIRECT_ONEHOT_COLUMNS = [
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Relationship_Status",
    "age_group",  # engineered categorical from engineering.py
]

# Bucketed before one-hot encoding — high cardinality relative to
# dataset size (110 countries / 705 rows).
BUCKETED_COLUMNS = ["Country"]
COUNTRY_TOP_N = 8


class RareCategoryBucketer:
    """
    Fits on a training series: keeps the `top_n` most frequent values
    as-is, maps everything else (including any category never seen in
    training) to `OTHER_LABEL`.
    """

    def __init__(self, top_n: int = COUNTRY_TOP_N):
        self.top_n = top_n
        self.categories_to_keep_: Optional[list[str]] = None

    def fit(self, series: pd.Series) -> "RareCategoryBucketer":
        self.categories_to_keep_ = (
            series.value_counts().head(self.top_n).index.tolist()
        )
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        if self.categories_to_keep_ is None:
            raise RuntimeError("Call .fit(train_series) before .transform().")
        return series.where(series.isin(self.categories_to_keep_), OTHER_LABEL)

    def fit_transform(self, series: pd.Series) -> pd.Series:
        return self.fit(series).transform(series)


class CategoricalFeatureEncoder:
    """
    Wraps one RareCategoryBucketer (for Country) + one sklearn
    OneHotEncoder (for every categorical column) behind a single
    fit/transform pair, so Phase 3 training code only has one object
    to fit on the train split and reuse on val/test.
    """

    def __init__(self):
        self.bucketer = RareCategoryBucketer()
        self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self._onehot_input_columns = DIRECT_ONEHOT_COLUMNS + BUCKETED_COLUMNS

    def fit(self, df: pd.DataFrame) -> "CategoricalFeatureEncoder":
        self.bucketer.fit(df["Country"])
        prepped = df.copy()
        prepped["Country"] = self.bucketer.transform(prepped["Country"])
        self.onehot.fit(prepped[self._onehot_input_columns])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        prepped = df.copy()
        prepped["Country"] = self.bucketer.transform(prepped["Country"])
        encoded = self.onehot.transform(prepped[self._onehot_input_columns])
        encoded_cols = self.onehot.get_feature_names_out(self._onehot_input_columns)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        remaining = df.drop(columns=self._onehot_input_columns)
        return pd.concat([remaining, encoded_df], axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
