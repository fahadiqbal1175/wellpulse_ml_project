"""
Phase 5 — Error Analysis (Section 13, second half): residual plots,
subgroup performance (country, academic level), and a manual review
of the 10 worst-predicted test rows.

Runs against the SAME final model + test predictions produced by
`src.evaluation.final_evaluation.build_final_eval_bundle()` — no
separate model fitting happens here, so there's exactly one source of
truth for "what did the final model predict on test."

Usage:
    python -m src.evaluation.error_analysis
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — this script only saves figures, never shows them
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.final_evaluation import FinalEvalBundle, build_final_eval_bundle
from src.models.baseline import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
COUNTRY_SUBGROUP_CSV = PROJECT_ROOT / "reports" / "phase5_subgroup_by_country.csv"
ACADEMIC_SUBGROUP_CSV = PROJECT_ROOT / "reports" / "phase5_subgroup_by_academic_level.csv"
WORST_PREDICTIONS_CSV = PROJECT_ROOT / "reports" / "phase5_worst_predictions.csv"

# The features called out in Phase 2/Section 9 as the main engineered
# signals — these are the ones worth checking for residual patterns
# (a model systematically over/under-predicting as a feature grows).
MAJOR_FEATURES = [
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Conflicts_Over_Social_Media",
    "usage_to_sleep_ratio",
]

# Countries with very few test rows produce noisy per-country MAE
# (e.g. n=1 means MAE *is* that one row's error) — reported but
# flagged rather than hidden, matching Section 13's ask to catch
# "a platform category with few examples"-style patterns.
MIN_SUBGROUP_N_FOR_RELIABLE_COMPARISON = 5


def compute_residuals(y_true: pd.Series, y_pred) -> pd.Series:
    """Residual = actual - predicted (positive = model under-predicted)."""
    return pd.Series(y_true.values - y_pred, index=y_true.index, name="residual")


def plot_predicted_vs_actual(y_true: pd.Series, y_pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
    lims = [min(y_true.min(), y_pred.min()) - 0.3, max(y_true.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Mental_Health_Score")
    ax.set_ylabel("Predicted Mental_Health_Score")
    ax.set_title("Predicted vs. Actual (test set)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals_vs_feature(residuals: pd.Series, feature_values: pd.Series, feature_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(feature_values, residuals, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"Residuals vs. {feature_name}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def subgroup_performance(test_df_raw: pd.DataFrame, y_true: pd.Series, y_pred, group_col: str) -> pd.DataFrame:
    """MAE/RMSE/R2 + row count per value of `group_col` (Section 13:
    'performance by subgroup ... to catch uneven performance across
    groups'). Uses the RAW (unbucketed) column so every real country
    shows up individually, not collapsed into the encoder's 'Other'."""
    rows = []
    for group_value, idx in test_df_raw.groupby(group_col).groups.items():
        yt = y_true.loc[idx]
        yp = pd.Series(y_pred, index=y_true.index).loc[idx]
        with warnings.catch_warnings():
            # R2 is undefined for n<2 (e.g. a country with 1 test row) —
            # expected for small subgroups, already flagged via
            # `reliable_n` below rather than worth a console warning.
            warnings.simplefilter("ignore")
            metrics = evaluate(yt, yp)
        rows.append({group_col: group_value, "n_rows": len(idx), **metrics})
    out = pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)
    out["reliable_n"] = out["n_rows"] >= MIN_SUBGROUP_N_FOR_RELIABLE_COMPARISON
    return out


def worst_predictions(test_df_raw: pd.DataFrame, y_true: pd.Series, y_pred, n: int = 10) -> pd.DataFrame:
    """The n rows with the largest absolute error, for manual review
    (Section 13: 'manually inspect the 10 worst-predicted rows and
    note any pattern')."""
    review_cols = [
        "Country", "Academic_Level", "Age", "Gender",
        "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night",
        "Conflicts_Over_Social_Media", "Most_Used_Platform", "Relationship_Status",
    ]
    out = test_df_raw[review_cols].copy()
    out["actual_score"] = y_true.values
    out["predicted_score"] = y_pred
    out["abs_error"] = (out["actual_score"] - out["predicted_score"]).abs()
    return out.sort_values("abs_error", ascending=False).head(n).reset_index(drop=True)


def run_error_analysis(bundle: FinalEvalBundle) -> dict:
    preds = bundle.final_model.predict(bundle.X_test)
    residuals = compute_residuals(bundle.y_test, preds)

    plot_predicted_vs_actual(bundle.y_test, preds, FIGURES_DIR / "phase5_predicted_vs_actual.png")
    for feature in MAJOR_FEATURES:
        plot_residuals_vs_feature(
            residuals, bundle.test_df_raw[feature], feature,
            FIGURES_DIR / f"phase5_residuals_vs_{feature}.png",
        )

    country_subgroup = subgroup_performance(bundle.test_df_raw, bundle.y_test, preds, "Country")
    academic_subgroup = subgroup_performance(bundle.test_df_raw, bundle.y_test, preds, "Academic_Level")
    worst = worst_predictions(bundle.test_df_raw, bundle.y_test, preds, n=10)

    return {
        "country_subgroup": country_subgroup,
        "academic_subgroup": academic_subgroup,
        "worst_predictions": worst,
    }


def main() -> None:
    bundle = build_final_eval_bundle()
    print(f"Running error analysis on final model: {bundle.final_model_name}\n")

    results = run_error_analysis(bundle)

    print(f"Saved predicted-vs-actual and {len(MAJOR_FEATURES)} residual-vs-feature plots -> {FIGURES_DIR}/")

    print("\n=== Subgroup Performance by Academic_Level (test set) ===")
    print(results["academic_subgroup"].to_string(index=False))
    ACADEMIC_SUBGROUP_CSV.parent.mkdir(parents=True, exist_ok=True)
    results["academic_subgroup"].to_csv(ACADEMIC_SUBGROUP_CSV, index=False)
    print(f"Saved to {ACADEMIC_SUBGROUP_CSV}")

    print("\n=== Subgroup Performance by Country (test set, top 10 by n_rows) ===")
    print(results["country_subgroup"].head(10).to_string(index=False))
    results["country_subgroup"].to_csv(COUNTRY_SUBGROUP_CSV, index=False)
    print(f"Saved full table to {COUNTRY_SUBGROUP_CSV}")

    print("\n=== 10 Worst-Predicted Test Rows ===")
    print(results["worst_predictions"].to_string(index=False))
    results["worst_predictions"].to_csv(WORST_PREDICTIONS_CSV, index=False)
    print(f"Saved to {WORST_PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
