"""
Phase 5 — Final Evaluation (Section 13, first half): the FIRST and
ONLY code path in this project that touches `split.test`.

Per `docs/PHASE4_TUNING.md`'s explicit next-step: Phase 4 did not
produce a single clear winner (untuned Decision Tree led on val MAE;
tuned Random Forest led on val RMSE/R2 and was the more
CV-robust result), so Phase 5 evaluates BOTH candidates against the
held-out test fold and lets that — not another val-set comparison —
decide the final model.

Candidates:
  - "Decision Tree (untuned)"   — Phase 3's leaderboard winner by MAE,
    refit fresh here (deterministic, same random_state, never saved
    as a file since Phase 3 only reported metrics).
  - "Random Forest (tuned)"     — Phase 4's best CV-validated model,
    loaded from models/phase4_best_model.joblib (already fit on the
    train fold with its tuned hyperparameters).

Decision rule: matches Section 11 / Section 13 — the blueprint's
stated PRIMARY metric is MAE, so the candidate with the lower TEST
MAE is selected as the final model. RMSE/R2 are still reported in
full for both, so the trade-off is visible rather than hidden (same
practice as Phase 3/4's docs).

Usage:
    python -m src.evaluation.final_evaluation
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeRegressor

from src.data.schema import validate_raw_dataset
from src.data.split import stratified_split
from src.features.engineering import compute_risk_tier
from src.models.baseline import (
    RANDOM_STATE,
    RAW_CSV,
    build_model_ready_xy,
    evaluate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TUNED_MODEL_PATH = PROJECT_ROOT / "models" / "phase4_best_model.joblib"
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
FINAL_EVAL_CSV = PROJECT_ROOT / "reports" / "phase5_final_evaluation.csv"
RISK_TIER_REPORT_CSV = PROJECT_ROOT / "reports" / "phase5_risk_tier_report.csv"

RISK_TIERS = ["low_risk", "medium_risk", "high_risk"]


@dataclass
class FinalEvalBundle:
    """Everything Phase 5's other modules (error_analysis, explainability)
    need, computed once here so all three modules agree on the exact
    same split/encoder/predictions instead of quietly re-deriving them."""

    test_df_raw: pd.DataFrame  # post-feature-engineering, pre-encoding (for subgroup/SHAP readability)
    X_test: pd.DataFrame
    y_test: pd.Series
    encoder: object
    candidates: dict[str, object]  # name -> fitted estimator
    final_model_name: str
    final_model: object
    leaderboard: pd.DataFrame  # test-set metrics for both candidates


def _add_engineered_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """Same per-row feature engineering `build_model_ready_xy` applies
    internally, exposed here so callers can inspect raw column values
    (Country, Academic_Level, etc.) alongside engineered ones, which
    `build_model_ready_xy`'s one-hot-encoded output no longer has."""
    from src.features.engineering import add_engineered_features
    from src.models.baseline import LEAKAGE_AND_ID_COLS

    return add_engineered_features(df).drop(columns=LEAKAGE_AND_ID_COLS)


def build_final_eval_bundle() -> FinalEvalBundle:
    """Reproduces the exact Phase 3/4 train/val/test split (same
    random_state=42), refits the untuned Decision Tree, loads the
    tuned Random Forest, and evaluates both on the test fold — the
    first time `split.test` is used anywhere in this project."""
    df = validate_raw_dataset(pd.read_csv(RAW_CSV))
    split = stratified_split(df)

    X_train, y_train, encoder = build_model_ready_xy(split.train, encoder=None, fit=True)
    X_test, y_test, _ = build_model_ready_xy(split.test, encoder=encoder, fit=False)

    decision_tree = DecisionTreeRegressor(random_state=RANDOM_STATE)
    decision_tree.fit(X_train, y_train)

    if not TUNED_MODEL_PATH.exists():
        raise SystemExit(
            f"{TUNED_MODEL_PATH} not found — run `python -m src.models.tuning` first."
        )
    tuned_rf = joblib.load(TUNED_MODEL_PATH)

    candidates = {
        "Decision Tree (untuned)": decision_tree,
        "Random Forest (tuned)": tuned_rf,
    }

    rows = []
    for name, model in candidates.items():
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        rows.append({"model": name, **metrics})
    leaderboard = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    final_model_name = leaderboard.iloc[0]["model"]
    final_model = candidates[final_model_name]

    test_df_raw = _add_engineered_features_only(split.test)

    return FinalEvalBundle(
        test_df_raw=test_df_raw,
        X_test=X_test,
        y_test=y_test,
        encoder=encoder,
        candidates=candidates,
        final_model_name=final_model_name,
        final_model=final_model,
        leaderboard=leaderboard,
    )


def risk_tier_classification_report(y_true: pd.Series, y_pred) -> pd.DataFrame:
    """Precision/recall/F1 per derived risk tier (Section 13: 'also
    reported: performance on the derived risk tiers ... so the
    UI-facing behavior is checked, not just the raw regression
    score'). Both true and predicted scores are mapped through the
    same `compute_risk_tier()` used everywhere else in the project."""
    true_tier = compute_risk_tier(pd.DataFrame({"Mental_Health_Score": y_true.values}))
    pred_tier = compute_risk_tier(pd.DataFrame({"Mental_Health_Score": y_pred}))

    precision, recall, f1, support = precision_recall_fscore_support(
        true_tier, pred_tier, labels=RISK_TIERS, zero_division=0
    )
    return pd.DataFrame(
        {
            "risk_tier": RISK_TIERS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def main() -> None:
    bundle = build_final_eval_bundle()

    print(f"train/val/test sizes reproduced: test={len(bundle.y_test)} rows")
    print("(test fold touched for the first time in this project — Phase 5 only)\n")

    print("=== Phase 5 Final Evaluation (TEST set, first and only look) ===")
    print(bundle.leaderboard.to_string(index=False))

    FINAL_EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    bundle.leaderboard.to_csv(FINAL_EVAL_CSV, index=False)
    print(f"\nSaved to {FINAL_EVAL_CSV}")

    print(f"\nSelected FINAL MODEL (lowest test MAE, the blueprint's primary metric): "
          f"{bundle.final_model_name}")

    final_preds = bundle.final_model.predict(bundle.X_test)
    tier_report = risk_tier_classification_report(bundle.y_test, final_preds)
    print("\n=== Risk-Tier Classification Report (final model, test set) ===")
    print(tier_report.to_string(index=False))
    tier_report.to_csv(RISK_TIER_REPORT_CSV, index=False)
    print(f"\nSaved to {RISK_TIER_REPORT_CSV}")

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle.final_model,
            "model_name": bundle.final_model_name,
            "encoder": bundle.encoder,
            "feature_columns": list(bundle.X_test.columns),
        },
        FINAL_MODEL_PATH,
    )
    print(f"Saved final model bundle (model + encoder + feature_columns) -> {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
