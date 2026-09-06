"""
Phase 4 — Advanced Modeling & Tuning (part 2): RandomizedSearchCV over
the top-N candidates from the Phase 4 leaderboard
(reports/phase4_leaderboard.csv), selected fresh each run rather than
hardcoded.

CV runs on the TRAIN fold only (5-fold on ~493 rows); val is scored
once at the end for a clean before/after comparison against Phase 3's
best untuned MAE; test stays untouched for Phase 5.

Every CV trial (not just the winner) is logged via
`src.experiments.logger.ExperimentLogger`, in a format Phase 6's
MLflow work is meant to read from / eventually replace.

Usage:
    python -m src.models.tuning
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV

from src.data.schema import validate_raw_dataset
from src.data.split import stratified_split
from src.experiments.logger import ExperimentLogger
from src.models.baseline import RANDOM_STATE, RAW_CSV, build_model_ready_xy, evaluate
from src.models.param_spaces import PARAM_SPACES, TUNING_MODEL_FACTORIES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE4_LEADERBOARD_CSV = PROJECT_ROOT / "reports" / "phase4_leaderboard.csv"
TUNING_SUMMARY_CSV = PROJECT_ROOT / "reports" / "phase4_tuning_summary.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "phase4_best_model.joblib"

# Phase 3's best untuned MAE (Decision Tree, docs/BASELINE_RESULTS.md).
# Keep this in sync if Phase 3's numbers are ever regenerated.
BASELINE_MAE = 0.150943962264151
TOP_N_CANDIDATES = 2
N_ITER = 60
CV_FOLDS = 5


def select_tuning_candidates(leaderboard: pd.DataFrame, top_n: int = TOP_N_CANDIDATES) -> list[str]:
    """
    Picks the top-N models BY MAE that also have a defined
    hyperparameter search space — computed fresh from the leaderboard
    each run rather than hardcoded, since adding LightGBM/XGBoost could
    change which two candidates actually lead.
    """
    ranked = leaderboard.sort_values("MAE")["model"].tolist()
    return [m for m in ranked if m in PARAM_SPACES][:top_n]


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    logger: ExperimentLogger,
) -> dict:
    if model_name not in PARAM_SPACES:
        raise ValueError(f"No param space defined for {model_name!r}")

    base_model = TUNING_MODEL_FACTORIES[model_name]()
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_SPACES[model_name],
        n_iter=N_ITER,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_)
    for _, row in cv_results.iterrows():
        logger.log_run(
            model_name=model_name,
            params=row["params"],
            metrics={"cv_MAE": -row["mean_test_score"], "cv_MAE_std": row["std_test_score"]},
            tags={"phase": "phase4_tuning", "stage": "cv_trial"},
        )

    best_model = search.best_estimator_
    val_metrics = evaluate(y_val, best_model.predict(X_val))
    val_metrics["best_cv_MAE"] = -search.best_score_

    logger.log_run(
        model_name=model_name,
        params=search.best_params_,
        metrics=val_metrics,
        tags={"phase": "phase4_tuning", "stage": "best_of_search"},
    )

    return {
        "model_name": model_name,
        "best_params": search.best_params_,
        "best_estimator": best_model,
        **val_metrics,
    }


def main() -> None:
    if not PHASE4_LEADERBOARD_CSV.exists():
        raise SystemExit(
            f"{PHASE4_LEADERBOARD_CSV} not found — run `python -m src.models.advanced` first."
        )

    leaderboard = pd.read_csv(PHASE4_LEADERBOARD_CSV)
    candidates = select_tuning_candidates(leaderboard)
    print(f"Tuning candidates (top {TOP_N_CANDIDATES} by val MAE with a defined param space): {candidates}")

    df = validate_raw_dataset(pd.read_csv(RAW_CSV))
    split = stratified_split(df)
    X_train, y_train, encoder = build_model_ready_xy(split.train, encoder=None, fit=True)
    X_val, y_val, _ = build_model_ready_xy(split.val, encoder=encoder, fit=False)

    logger = ExperimentLogger(experiment="phase4_tuning")

    summary_rows = []
    best_overall = None
    for model_name in candidates:
        print(f"\n--- Tuning {model_name} ({N_ITER} iters x {CV_FOLDS}-fold CV) ---")
        result = tune_model(model_name, X_train, y_train, X_val, y_val, logger)
        print(
            f"{model_name}: val MAE={result['MAE']:.4f}  val RMSE={result['RMSE']:.4f}  "
            f"val R2={result['R2']:.4f}  (best CV MAE={result['best_cv_MAE']:.4f})"
        )

        summary_rows.append(
            {
                "model": model_name,
                "val_MAE": result["MAE"],
                "val_RMSE": result["RMSE"],
                "val_R2": result["R2"],
                "best_cv_MAE": result["best_cv_MAE"],
                "best_params": result["best_params"],
            }
        )
        if best_overall is None or result["MAE"] < best_overall["MAE"]:
            best_overall = result

    summary_df = pd.DataFrame(summary_rows).sort_values("val_MAE").reset_index(drop=True)
    summary_df.to_csv(TUNING_SUMMARY_CSV, index=False)
    print(f"\nSaved tuning summary -> {TUNING_SUMMARY_CSV}")

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_overall["best_estimator"], BEST_MODEL_PATH)
    print(f"Saved best tuned model -> {BEST_MODEL_PATH}")

    print("\n=== Verification ===")
    print(f"Best untuned baseline (Phase 3, MAE):                {BASELINE_MAE:.4f}")
    print(f"Best tuned model ({best_overall['model_name']}, val MAE):        {best_overall['MAE']:.4f}")
    if best_overall["MAE"] < BASELINE_MAE:
        improvement = (BASELINE_MAE - best_overall["MAE"]) / BASELINE_MAE * 100
        print(f"PASS -- tuned model improves MAE by {improvement:.1f}%")
    else:
        print("FAIL -- tuning did not beat the untuned baseline; see docs/PHASE4_TUNING.md for next steps")


if __name__ == "__main__":
    main()
