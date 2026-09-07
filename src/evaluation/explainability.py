"""
Phase 5 — Explainable AI (Section 14): SHAP is kept, with its role
made explicit and separated from the model itself:

    Prediction (from the trained regressor)
    +
    A simple confidence proxy (prediction interval from the model's
    residual spread, measured on the VALIDATION set — the same set
    Phase 3/4 already used repeatedly, so this doesn't spend any of
    test's single-use budget)
    +
    Top 3 SHAP-contributing features for this specific prediction
    +
    A human-readable sentence generated from the top features
    (a template, not a model)

Both Phase 5 candidates (Decision Tree, Random Forest) are tree
models, so `shap.TreeExplainer` is used — exact and fast, no
approximation needed.

Limitations (Section 14, stated here and belongs in Section 31's
docs too): SHAP explains *this model's* reasoning, not biological/
psychological causation; a feature can be a top contributor without
being a true cause; explanations can shift if the model is retrained.

Usage:
    python -m src.evaluation.explainability
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.evaluation.final_evaluation import FinalEvalBundle, build_final_eval_bundle
from src.features.engineering import compute_risk_tier
from src.models.baseline import build_model_ready_xy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SAMPLE_EXPLANATIONS_JSON = PROJECT_ROOT / "reports" / "phase5_sample_explanations.json"

TOP_N_FACTORS = 3
CONFIDENCE_Z = 1.0  # ~68% interval under a normal residual assumption — simple and honest, not oversold as a calibrated interval


def _prettify_feature_name(name: str) -> str:
    """Turns an encoded/engineered column name into a readable phrase
    for the templated sentence, e.g. 'Most_Used_Platform_Instagram'
    -> 'mostly using Instagram', 'usage_to_sleep_ratio' -> 'usage-to-
    sleep ratio'. Falls back to underscore-to-space for anything else."""
    onehot_prefixes = {
        "Gender_": "being {}",
        "Academic_Level_": "being a {} student",
        "Most_Used_Platform_": "mostly using {}",
        "Relationship_Status_": "relationship status ({})",
        "age_group_": "age group {}",
        "Country_": "being based in {}",
    }
    if name == "Country_Other":
        # RareCategoryBucketer's catch-all (src/features/encoding.py) —
        # "Other" isn't a real country, so name it as the bucket it is.
        return "being from a less-common country in this dataset"

    for prefix, template in onehot_prefixes.items():
        if name.startswith(prefix):
            value = name[len(prefix):]
            return template.format(value)

    manual = {
        "usage_to_sleep_ratio": "usage-to-sleep ratio",
        "usage_conflict_interaction": "usage-conflict interaction",
        "Avg_Daily_Usage_Hours": "daily social-media usage hours",
        "Sleep_Hours_Per_Night": "sleep hours per night",
        "Conflicts_Over_Social_Media": "conflicts over social media",
        "Age": "age",
    }
    return manual.get(name, name.replace("_", " "))


def compute_residual_std_on_val() -> float:
    """The 'model's residual spread' referenced in Section 14, measured
    on VAL (already used repeatedly in Phases 3-4) rather than test —
    test is reserved for the headline generalization numbers computed
    once in `final_evaluation.py`, not spent again here."""
    from src.data.schema import validate_raw_dataset
    from src.data.split import stratified_split
    from src.models.baseline import RAW_CSV, evaluate

    bundle = build_final_eval_bundle()
    df = validate_raw_dataset(pd.read_csv(RAW_CSV))
    split = stratified_split(df)
    X_val, y_val, _ = build_model_ready_xy(split.val, encoder=bundle.encoder, fit=False)
    val_preds = bundle.final_model.predict(X_val)
    residuals = y_val.values - val_preds
    return float(np.std(residuals))


def compute_shap_values(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def top_n_factors(shap_row, feature_names: list[str], n: int = TOP_N_FACTORS) -> list[dict]:
    """Ranks one prediction's SHAP contributions by absolute magnitude
    and returns the top `n` as {feature, pretty_name, shap_value, direction}."""
    order = np.argsort(-np.abs(shap_row))[:n]
    factors = []
    for i in order:
        value = float(shap_row[i])
        factors.append(
            {
                "feature": feature_names[i],
                "pretty_name": _prettify_feature_name(feature_names[i]),
                "shap_value": value,
                "direction": "raising" if value > 0 else "lowering",
            }
        )
    return factors


def generate_explanation_sentence(factors: list[dict]) -> str:
    """A template, not a model (Section 14/15's explicit separation)."""
    phrases = [f"{f['pretty_name']} ({f['direction']} the score)" for f in factors]
    if len(phrases) == 1:
        joined = phrases[0]
    elif len(phrases) == 2:
        joined = f"{phrases[0]} and {phrases[1]}"
    else:
        joined = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
    return f"This prediction was most shaped by {joined}."


def explain_single_prediction(
    row_index: int,
    bundle: FinalEvalBundle,
    shap_values,
    residual_std: float,
) -> dict:
    prediction = float(bundle.final_model.predict(bundle.X_test.iloc[[row_index]])[0])
    factors = top_n_factors(shap_values.values[row_index], list(bundle.X_test.columns))
    sentence = generate_explanation_sentence(factors)

    return {
        "test_row_index": int(bundle.X_test.index[row_index]),
        "actual_score": float(bundle.y_test.iloc[row_index]),
        "predicted_score": round(prediction, 3),
        "confidence_interval_68pct": [
            round(prediction - CONFIDENCE_Z * residual_std, 3),
            round(prediction + CONFIDENCE_Z * residual_std, 3),
        ],
        "top_factors": factors,
        "explanation_sentence": sentence,
    }


def pick_sample_rows(bundle: FinalEvalBundle) -> list[int]:
    """One row per risk tier (low/medium/high) so the 3 reviewed
    examples (Section 37's verify criterion) show varied behavior
    rather than 3 similar mid-range cases."""
    tiers = compute_risk_tier(pd.DataFrame({"Mental_Health_Score": bundle.y_test.values}))
    positions = []
    for tier in ["low_risk", "medium_risk", "high_risk"]:
        matches = np.where(tiers.values == tier)[0]
        if len(matches) > 0:
            positions.append(int(matches[0]))
    return positions


def main() -> None:
    bundle = build_final_eval_bundle()
    print(f"Explaining final model: {bundle.final_model_name}\n")

    shap_values = compute_shap_values(bundle.final_model, bundle.X_test)

    # For a RandomForestRegressor, shap.TreeExplainer returns values
    # for the ensemble's averaged output directly (no per-class axis
    # to squeeze) — this holds for both Phase 5 candidates.
    plt.figure()
    shap.summary_plot(shap_values, bundle.X_test, show=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase5_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot -> {FIGURES_DIR / 'phase5_shap_summary.png'}")

    residual_std = compute_residual_std_on_val()
    print(f"Residual std (val set, used as the confidence proxy spread): {residual_std:.4f}")

    sample_positions = pick_sample_rows(bundle)
    explanations = [
        explain_single_prediction(pos, bundle, shap_values, residual_std)
        for pos in sample_positions
    ]

    print(f"\n=== {len(explanations)} Sample Explanations (one per risk tier) ===")
    for exp in explanations:
        print(f"\nTest row {exp['test_row_index']}  "
              f"actual={exp['actual_score']}  predicted={exp['predicted_score']}  "
              f"68% interval={exp['confidence_interval_68pct']}")
        print(f"  {exp['explanation_sentence']}")

    SAMPLE_EXPLANATIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_EXPLANATIONS_JSON, "w") as f:
        json.dump(
            {
                "final_model_name": bundle.final_model_name,
                "residual_std_val": residual_std,
                "confidence_proxy_note": (
                    "Symmetric interval = prediction +/- 1 residual std, "
                    "measured on the validation set. This is a simple spread "
                    "proxy, not a calibrated statistical prediction interval."
                ),
                "explanations": explanations,
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {SAMPLE_EXPLANATIONS_JSON}")


if __name__ == "__main__":
    main()
