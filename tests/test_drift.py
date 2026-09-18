"""
Phase 13 — Monitoring & Drift tests (Section 29).

Covers compute_psi's known-value behavior, severity tier boundaries,
and Milestone ML-9's verify criterion directly: a deliberately
shifted synthetic dataframe must be flagged "significant" against a
small, hand-built reference — and, just as importantly, a batch that
actually matches the reference at a realistic window size (N=200)
must NOT be flagged, guarding against a drift job that cries wolf on
ordinary traffic.
"""
import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import build_drift_report, compute_psi, severity_tier


def test_compute_psi_identical_distributions_is_zero():
    props = [0.25, 0.25, 0.25, 0.25]
    assert compute_psi(props, props) == pytest.approx(0.0)


def test_compute_psi_detects_shift():
    ref = [0.5, 0.5]
    shifted = [0.05, 0.95]
    assert compute_psi(ref, shifted) > 0.25


@pytest.mark.parametrize(
    "psi, expected",
    [(0.0, "stable"), (0.05, "stable"), (0.15, "moderate"), (0.3, "significant")],
)
def test_severity_tier_boundaries(psi, expected):
    assert severity_tier(psi) == expected


@pytest.fixture
def small_reference():
    """Hand-built, mimicking build_reference_distribution's shape but
    small enough to reason about by hand."""
    return {
        "numeric_features": {
            "Sleep_Hours_Per_Night": {
                "bin_edges": [-np.inf, 4, 6, 8, np.inf],
                "reference_proportions": [0.1, 0.3, 0.4, 0.2],
            },
        },
        "categorical_features": {
            "Most_Used_Platform": {
                "known_categories": ["Instagram", "TikTok", "YouTube"],
                "reference_proportions": {
                    "Instagram": 0.5, "TikTok": 0.3, "YouTube": 0.2, "__unexpected__": 0.0,
                },
            },
        },
        "prediction": {
            "bin_edges": [-np.inf, 4, 6, 8, np.inf],
            "reference_proportions": [0.2, 0.3, 0.3, 0.2],
            "reference_mean": 5.5,
        },
    }


def test_deliberately_shifted_batch_triggers_significant_flag(small_reference):
    # Every row sleeps far less than any training respondent, and
    # every row uses a platform the reference has never seen.
    shifted = pd.DataFrame({
        "Sleep_Hours_Per_Night": [1.0] * 40,
        "Most_Used_Platform": ["LINE"] * 40,
        "predicted_score": [9.5] * 40,
    })
    report = build_drift_report(small_reference, shifted)
    assert report["verdict"] == "significant"
    assert report["worst_check"]["psi"] >= 0.25


def test_matching_batch_at_realistic_volume_stays_stable(small_reference):
    sleep_vals = [2] * 20 + [5] * 60 + [7] * 80 + [10] * 40
    platform_vals = ["Instagram"] * 100 + ["TikTok"] * 60 + ["YouTube"] * 40
    pred_vals = [3] * 40 + [5] * 60 + [7] * 60 + [9] * 40
    matching = pd.DataFrame({
        "Sleep_Hours_Per_Night": sleep_vals,
        "Most_Used_Platform": platform_vals,
        "predicted_score": pred_vals,
    })
    report = build_drift_report(small_reference, matching)
    assert report["verdict"] == "stable"