"""
Phase 6 — MLflow & Model Registry (Section 16/17), the actual entry
point: takes the Phase 5 final model, logs ONE comprehensive MLflow
run for it (metrics + all Phase 5 artifacts), registers it in the
Model Registry, sets a "Staging" alias, and decides whether to move
the "Production" alias to it per the blueprint's promotion criteria.

Does NOT recompute the split or refit either Phase 5 candidate
differently — it calls `build_final_eval_bundle()` (the same function
Phase 5's own scripts use), so this is guaranteed to log the exact
same model/predictions Phase 5 already reported, never a fresh
re-derivation that could quietly drift from those numbers.

Also re-runs Phase 5's own `error_analysis.main()` and
`explainability.main()` first, so the PNGs/CSVs/JSON this script
attaches as MLflow artifacts are freshly regenerated (not stale files
left over from a previous run) — safe to do because `random_state=42`
makes every one of those steps fully deterministic given the same
data and model.

Usage:
    python -m src.experiments.register_final_model
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor

from src.data.schema import validate_raw_dataset
from src.data.split import stratified_split
from src.evaluation import error_analysis, explainability
from src.evaluation.final_evaluation import (
    FINAL_EVAL_CSV,
    FINAL_MODEL_PATH,
    RISK_TIER_REPORT_CSV,
    build_final_eval_bundle,
    risk_tier_classification_report,
)
from src.experiments.mlflow_utils import (
    REGISTERED_MODEL_NAME,
    log_model_run,
    register_and_maybe_promote,
    standard_tags,
)
from src.models.baseline import RAW_CSV, build_model_ready_xy, evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DECISION_RECORD_PATH = PROJECT_ROOT / "reports" / "phase6_registry_decision.json"

MEANINGFUL_MARGIN_PCT = 20.0

# Every one of these must exist after error_analysis.main() and
# explainability.main() have run — they're attached as MLflow
# artifacts alongside the model itself.
PHASE5_ARTIFACT_PATHS = [
    FIGURES_DIR / "phase5_predicted_vs_actual.png",
    FIGURES_DIR / "phase5_residuals_vs_Avg_Daily_Usage_Hours.png",
    FIGURES_DIR / "phase5_residuals_vs_Sleep_Hours_Per_Night.png",
    FIGURES_DIR / "phase5_residuals_vs_Conflicts_Over_Social_Media.png",
    FIGURES_DIR / "phase5_residuals_vs_usage_to_sleep_ratio.png",
    FIGURES_DIR / "phase5_shap_summary.png",
    PROJECT_ROOT / "reports" / "phase5_subgroup_by_country.csv",
    PROJECT_ROOT / "reports" / "phase5_subgroup_by_academic_level.csv",
    PROJECT_ROOT / "reports" / "phase5_worst_predictions.csv",
    PROJECT_ROOT / "reports" / "phase5_sample_explanations.json",
    FINAL_EVAL_CSV,
    RISK_TIER_REPORT_CSV,
]


def compute_mean_baseline_test_mae(bundle) -> float:
    """A fresh DummyRegressor(mean), fit on the SAME train fold and
    scored on the SAME test fold as `bundle` — the "mean baseline"
    comparison point Section 17 calls for when there's no existing
    Production model to compare against yet. Reuses `bundle.encoder`
    (already fit on train inside `build_final_eval_bundle`) rather
    than fitting a second, potentially-different encoder."""
    df = validate_raw_dataset(pd.read_csv(RAW_CSV))
    split = stratified_split(df)
    X_train, y_train, _ = build_model_ready_xy(split.train, encoder=bundle.encoder, fit=False)

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    preds = dummy.predict(bundle.X_test)
    return evaluate(bundle.y_test, preds)["MAE"]


def flatten_risk_tier_metrics(tier_report: pd.DataFrame) -> dict:
    """{'risktier_low_risk_precision': ..., 'risktier_low_risk_recall':
    ..., ...} — flattened so each precision/recall/f1/support cell
    becomes its own loggable MLflow metric."""
    flat = {}
    for _, row in tier_report.iterrows():
        tier = row["risk_tier"]
        for col in ("precision", "recall", "f1", "support"):
            flat[f"risktier_{tier}_{col}"] = row[col]
    return flat


def main() -> None:
    bundle = build_final_eval_bundle()
    print(f"Final model (from Phase 5): {bundle.final_model_name}")

    print("\nRegenerating Phase 5 artifacts (error analysis + SHAP explainability)...")
    error_analysis.main()
    explainability.main()

    for path in PHASE5_ARTIFACT_PATHS:
        if not path.exists():
            raise SystemExit(f"Expected Phase 5 artifact missing: {path}")

    final_preds = bundle.final_model.predict(bundle.X_test)
    test_metrics = evaluate(bundle.y_test, final_preds)
    tier_report = risk_tier_classification_report(bundle.y_test, final_preds)
    tier_metrics = flatten_risk_tier_metrics(tier_report)

    mean_baseline_test_mae = compute_mean_baseline_test_mae(bundle)
    print(f"\nFresh DummyRegressor(mean) test MAE (same train/test fold): {mean_baseline_test_mae:.4f}")

    metrics = {
        "test_MAE": test_metrics["MAE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_R2": test_metrics["R2"],
        "mean_baseline_test_MAE": mean_baseline_test_mae,
        **tier_metrics,
    }

    params = (
        bundle.final_model.get_params()
        if hasattr(bundle.final_model, "get_params")
        else {}
    )
    tags = standard_tags(
        phase="phase6_registry",
        stage="final_model",
        model_type=bundle.final_model_name,
    )

    run_id = log_model_run(
        run_name=f"phase6_final_{bundle.final_model_name}",
        model=bundle.final_model,
        params=params,
        metrics=metrics,
        tags=tags,
        extra_artifacts=[str(FINAL_MODEL_PATH), *[str(p) for p in PHASE5_ARTIFACT_PATHS]],
        input_example=bundle.X_test.head(2),
    )
    print(f"\nLogged final-model MLflow run: {run_id}")

    decision = register_and_maybe_promote(
        run_id=run_id,
        model_name=REGISTERED_MODEL_NAME,
        test_mae=test_metrics["MAE"],
        mean_baseline_test_mae=mean_baseline_test_mae,
        meaningful_margin_pct=MEANINGFUL_MARGIN_PCT,
    )

    print("\n=== Phase 6 Registry Decision ===")
    for key, value in decision.items():
        print(f"  {key}: {value}")

    DECISION_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISION_RECORD_PATH, "w") as f:
        json.dump(decision, f, indent=2, default=str)
    print(f"\nSaved decision record -> {DECISION_RECORD_PATH}")


if __name__ == "__main__":
    main()
