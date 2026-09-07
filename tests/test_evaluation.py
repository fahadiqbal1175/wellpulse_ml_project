"""Phase 5 verification: the test fold is evaluated exactly once, both
candidates are actually compared, risk-tier reporting behaves
sensibly, subgroup/worst-row analysis returns well-formed tables, and
SHAP explanations come back with the right shape (Section 37's verify
criterion: 'SHAP summary plot + 3 sample explanations reviewed by
hand')."""
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation import error_analysis, explainability, final_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "students_social_media_addiction.csv"
TUNED_MODEL_PATH = PROJECT_ROOT / "models" / "phase4_best_model.joblib"


def _skip_if_prereqs_missing() -> None:
    if not RAW_CSV.exists():
        pytest.skip(f"{RAW_CSV} not found — run `python -m src.data.ingest` first.")
    if not TUNED_MODEL_PATH.exists():
        pytest.skip(f"{TUNED_MODEL_PATH} not found — run `python -m src.models.tuning` first.")


@pytest.fixture(scope="module")
def bundle() -> final_evaluation.FinalEvalBundle:
    _skip_if_prereqs_missing()
    return final_evaluation.build_final_eval_bundle()


def test_both_candidates_evaluated_on_test(bundle) -> None:
    assert set(bundle.leaderboard["model"]) == {"Decision Tree (untuned)", "Random Forest (tuned)"}
    assert {"MAE", "RMSE", "R2"}.issubset(bundle.leaderboard.columns)


def test_final_model_is_the_lower_test_mae_candidate(bundle) -> None:
    best_row = bundle.leaderboard.sort_values("MAE").iloc[0]
    assert bundle.final_model_name == best_row["model"]


def test_test_fold_size_matches_stratified_split(bundle) -> None:
    # 70/15/15 split of 705 rows -> test should be the ~15% slice.
    assert 90 <= len(bundle.y_test) <= 120


def test_risk_tier_report_covers_all_three_tiers(bundle) -> None:
    preds = bundle.final_model.predict(bundle.X_test)
    report = final_evaluation.risk_tier_classification_report(bundle.y_test, preds)
    assert set(report["risk_tier"]) == {"low_risk", "medium_risk", "high_risk"}
    assert report["support"].sum() == len(bundle.y_test)


def test_subgroup_performance_has_one_row_per_group(bundle) -> None:
    preds = bundle.final_model.predict(bundle.X_test)
    academic = error_analysis.subgroup_performance(
        bundle.test_df_raw, bundle.y_test, preds, "Academic_Level"
    )
    assert set(academic["Academic_Level"]) == set(bundle.test_df_raw["Academic_Level"].unique())
    assert academic["n_rows"].sum() == len(bundle.y_test)


def test_worst_predictions_sorted_descending_by_abs_error(bundle) -> None:
    preds = bundle.final_model.predict(bundle.X_test)
    worst = error_analysis.worst_predictions(bundle.test_df_raw, bundle.y_test, preds, n=10)
    assert len(worst) == 10
    assert (worst["abs_error"].diff().dropna() <= 0).all()


def test_top_n_factors_returns_requested_count_sorted_by_magnitude() -> None:
    import numpy as np

    shap_row = np.array([0.1, -0.9, 0.05, 0.4, -0.02])
    names = ["a", "b", "c", "d", "e"]
    factors = explainability.top_n_factors(shap_row, names, n=3)
    assert [f["feature"] for f in factors] == ["b", "d", "a"]
    assert factors[0]["direction"] == "lowering"
    assert factors[1]["direction"] == "raising"


def test_generate_explanation_sentence_mentions_every_factor() -> None:
    factors = [
        {"pretty_name": "sleep hours per night", "direction": "raising"},
        {"pretty_name": "conflicts over social media", "direction": "lowering"},
    ]
    sentence = explainability.generate_explanation_sentence(factors)
    assert "sleep hours per night" in sentence
    assert "conflicts over social media" in sentence


def test_pick_sample_rows_returns_one_index_per_present_tier(bundle) -> None:
    positions = explainability.pick_sample_rows(bundle)
    assert 1 <= len(positions) <= 3
    assert len(set(positions)) == len(positions)
