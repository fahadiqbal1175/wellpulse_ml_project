"""
Phase 3 — Baseline Models (Section 11, first half): mean/median
baseline through Random Forest, no hyperparameter tuning yet (that's
Phase 4). Produces the leaderboard table that's this phase's
verification artifact (Section 37) and satisfies Milestone ML-3
("at least 4 model families are trained and compared on the same
split").

Model comparison uses VAL only — the test fold from
`src.data.split.stratified_split()` is deliberately never touched
here, so it stays a clean, single-use estimate for Phase 5's final
evaluation (see the module docstring in `src/data/split.py`).

Usage:
    python -m src.models.baseline
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from src.data.schema import validate_raw_dataset
from src.data.split import country_generalization_split, stratified_split
from src.features.encoding import CategoricalFeatureEncoder
from src.features.engineering import add_engineered_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "students_social_media_addiction.csv"
LEADERBOARD_CSV = PROJECT_ROOT / "reports" / "phase3_leaderboard.csv"
COUNTRY_CHECK_CSV = PROJECT_ROOT / "reports" / "phase3_country_generalization.csv"

TARGET_COL = "Mental_Health_Score"
# Excluded per the Phase 2 leakage check (docs/FEATURES.md) and
# because Student_ID carries no predictive meaning.
LEAKAGE_AND_ID_COLS = ["Addicted_Score", "Affects_Academic_Performance", "Student_ID"]

RANDOM_STATE = 42

# Every entry here is a distinct, freshly-constructed model — no
# hyperparameter search yet (Phase 4's job). Ridge/Lasso are included
# per Section 11's "Linear Regression (+ Ridge/Lasso)" step.
MODEL_FACTORIES = {
    "Dummy (mean)": lambda: DummyRegressor(strategy="mean"),
    "Dummy (median)": lambda: DummyRegressor(strategy="median"),
    "Linear Regression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(random_state=RANDOM_STATE),
    "Lasso": lambda: Lasso(random_state=RANDOM_STATE),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE
    ),
}


def build_model_ready_xy(df: pd.DataFrame, encoder: CategoricalFeatureEncoder | None, fit: bool):
    """
    Applies engineered features, drops leakage/ID columns, separates
    the target, and one-hot encodes categoricals — fitting the encoder
    if `fit=True` (train only), otherwise reusing an already-fit one.
    """
    engineered = add_engineered_features(df).drop(columns=LEAKAGE_AND_ID_COLS)
    y = engineered[TARGET_COL]
    X_raw = engineered.drop(columns=[TARGET_COL])

    if fit:
        encoder = CategoricalFeatureEncoder()
        X = encoder.fit_transform(X_raw)
    else:
        if encoder is None:
            raise ValueError("Must pass a fitted encoder when fit=False.")
        X = encoder.transform(X_raw)

    return X, y, encoder


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def run_baseline_leaderboard(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_factories: dict | None = None,
    return_models: bool = False,
):
    """
    `model_factories` defaults to Phase 3's MODEL_FACTORIES so existing
    callers (and tests) are unaffected. Phase 4 (src/models/advanced.py)
    passes an extended dict so LightGBM/XGBoost are trained and scored
    through this exact same loop rather than a duplicated one.

    `return_models` (Phase 6 addition, default False so every existing
    call site and test is unaffected): when True, returns
    `(leaderboard, fitted_models)` instead of just `leaderboard`, where
    `fitted_models` is `{name: fitted_estimator}` — added so the Phase
    6 MLflow logging step in baseline.py/advanced.py's `main()` can log
    each already-trained model without re-fitting all 7-9 of them a
    second time.
    """
    if model_factories is None:
        model_factories = MODEL_FACTORIES

    X_train, y_train, encoder = build_model_ready_xy(train_df, encoder=None, fit=True)
    X_val, y_val, _ = build_model_ready_xy(val_df, encoder=encoder, fit=False)

    rows = []
    fitted_models = {}
    for name, factory in model_factories.items():
        model = factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = evaluate(y_val, preds)
        rows.append({"model": name, **metrics})
        if return_models:
            fitted_models[name] = model

    leaderboard = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    if return_models:
        return leaderboard, fitted_models
    return leaderboard


def run_country_generalization_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supplementary diagnostic (Section 10): train Random Forest on a
    subset of countries, evaluate on entirely unseen countries, and
    compare against evaluating on an equally-sized RANDOM subset of
    the training countries' own held-back rows — an apples-to-apples
    comparison of "seen-country" vs "unseen-country" error, isolating
    the effect of the country itself rather than just sample size.
    """
    train_countries_df, holdout_countries_df = country_generalization_split(df)

    # Further split train_countries_df into a fit/eval portion so we
    # have a same-distribution (seen-country) comparison point of
    # matching size to the unseen-country holdout.
    from sklearn.model_selection import train_test_split

    fit_df, seen_eval_df = train_test_split(
        train_countries_df,
        test_size=len(holdout_countries_df),
        random_state=RANDOM_STATE,
    )

    X_fit, y_fit, encoder = build_model_ready_xy(fit_df, encoder=None, fit=True)
    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_fit, y_fit)

    X_seen, y_seen, _ = build_model_ready_xy(seen_eval_df, encoder=encoder, fit=False)
    X_unseen, y_unseen, _ = build_model_ready_xy(holdout_countries_df, encoder=encoder, fit=False)

    seen_metrics = evaluate(y_seen, model.predict(X_seen))
    unseen_metrics = evaluate(y_unseen, model.predict(X_unseen))

    result = pd.DataFrame(
        [
            {"eval_set": "seen_countries (held-back rows)", "n_rows": len(seen_eval_df), **seen_metrics},
            {"eval_set": "unseen_countries (fully held out)", "n_rows": len(holdout_countries_df), **unseen_metrics},
        ]
    )
    return result


def _log_leaderboard_to_mlflow(
    leaderboard: pd.DataFrame,
    fitted_models: dict,
    X_train: pd.DataFrame,
    phase: str,
) -> None:
    """Phase 6: logs every already-trained model in `fitted_models` as
    its own lightweight MLflow run (params + val metrics + model
    artifact), so `mlflow ui` shows full lineage across every phase —
    not just Phase 4's tuning trials (Milestone ML-4)."""
    from src.experiments.mlflow_utils import log_model_run, standard_tags

    for _, row in leaderboard.iterrows():
        name = row["model"]
        model = fitted_models[name]
        params = model.get_params() if hasattr(model, "get_params") else {}
        metrics = {"MAE": row["MAE"], "RMSE": row["RMSE"], "R2": row["R2"]}
        tags = standard_tags(phase=phase, stage="leaderboard", model_type=name)
        try:
            log_model_run(
                run_name=f"{phase}_{name}",
                model=model,
                params=params,
                metrics=metrics,
                tags=tags,
                input_example=X_train.head(2),
            )
        except Exception as exc:  # pragma: no cover - one bad model must never block the rest
            print(f"[MLflow logging] skipped {name}: {exc}")


def main() -> None:
    df = validate_raw_dataset(pd.read_csv(RAW_CSV))
    split = stratified_split(df)

    print(f"train/val/test sizes: {len(split.train)}/{len(split.val)}/{len(split.test)}")
    print("(test fold is held out untouched — not used in this leaderboard)")
    print()

    leaderboard, fitted_models = run_baseline_leaderboard(
        split.train, split.val, return_models=True
    )
    print("=== Phase 3 Baseline Leaderboard (val set) ===")
    print(leaderboard.to_string(index=False))

    LEADERBOARD_CSV.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(LEADERBOARD_CSV, index=False)
    print(f"\nSaved to {LEADERBOARD_CSV}")

    try:
        X_train, _, _ = build_model_ready_xy(split.train, encoder=None, fit=True)
        _log_leaderboard_to_mlflow(leaderboard, fitted_models, X_train, phase="phase3_baseline")
        print(f"Logged {len(fitted_models)} model runs to MLflow (experiment: wellpulse_mental_health_score)")
    except Exception as exc:  # pragma: no cover - MLflow logging is best-effort
        print(f"[main] MLflow logging skipped: {exc}")

    print()
    country_check = run_country_generalization_check(df)
    print("=== Country Generalization Check (Random Forest) ===")
    print(country_check.to_string(index=False))
    country_check.to_csv(COUNTRY_CHECK_CSV, index=False)
    print(f"\nSaved to {COUNTRY_CHECK_CSV}")


if __name__ == "__main__":
    main()
