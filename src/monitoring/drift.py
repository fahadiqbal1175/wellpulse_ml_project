"""
Phase 13 — Monitoring & Drift (Sections 25/26): PSI-based drift
computation. Pure functions, no I/O, no DB, no model — deliberately
independent of everything else so it's trivially unit-testable
(tests/test_drift.py) and imports nothing heavier than pandas/numpy.

Uses Population Stability Index (PSI) for every check — numeric
features (quantile-binned against the training distribution),
categorical features (category proportions), and the predicted score
itself — rather than splitting numeric features off onto a KS-test.
One metric, one set of thresholds, one thing to explain to an
interviewer. Section 25 lists PSI and KS-test as interchangeable
options; unifying on PSI also means this project never needs scipy.

Conventional PSI bands (Section 26):
    PSI < 0.10             -> "stable"      (no meaningful shift)
    0.10 <= PSI < 0.25      -> "moderate"    (worth a look)
    PSI >= 0.25             -> "significant" (flagged)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25
EPSILON = 1e-4


def severity_tier(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return "significant"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "stable"


def compute_psi(reference_proportions: list[float], live_proportions: list[float]) -> float:
    """sum((live% - ref%) * ln(live% / ref%)) per bucket. Both sides
    floored at EPSILON so an empty bucket on either side never divides
    by zero or takes log(0) — the price of that safety is that a truly
    empty bucket contributes a fixed, large-ish term rather than
    infinity (see reference.py's N_BINS=5 comment for why this is
    tuned to not fire on ordinary sampling noise)."""
    psi = 0.0
    for ref_p, live_p in zip(reference_proportions, live_proportions):
        ref_p = max(ref_p, EPSILON)
        live_p = max(live_p, EPSILON)
        psi += (live_p - ref_p) * math.log(live_p / ref_p)
    return psi


def bin_numeric(values: pd.Series, bin_edges: list[float]) -> list[float]:
    """Proportions of `values` falling into each bucket defined by
    `bin_edges` (edges include -inf/+inf at the ends, so an
    out-of-range live value still lands in the extreme bucket instead
    of being silently dropped from the comparison)."""
    counts = pd.cut(values, bins=bin_edges, include_lowest=True).value_counts(sort=False)
    total = counts.sum()
    if total == 0:
        return [0.0] * len(counts)
    return (counts / total).tolist()


def category_proportions(values: pd.Series, known_categories: list[str]) -> dict[str, float]:
    """Proportions over `known_categories` plus a catch-all
    '__unexpected__' bucket for anything not in that list — a live
    category the training data never contained shows up here as pure
    drift, by construction."""
    total = len(values)
    if total == 0:
        return {cat: 0.0 for cat in known_categories} | {"__unexpected__": 0.0}
    is_known = values.isin(known_categories)
    props = {cat: float((values == cat).sum()) / total for cat in known_categories}
    props["__unexpected__"] = float((~is_known).sum()) / total
    return props


def unexpected_category_rate(values: pd.Series, known_categories: list[str]) -> float:
    """Section 25's separate "missing/unexpected category rate"
    metric — reported alongside PSI, not folded into the verdict, so
    it's visible on its own even when PSI's epsilon-smoothing masks
    how large the raw rate actually is."""
    if len(values) == 0:
        return 0.0
    return float((~values.isin(known_categories)).sum()) / len(values)


def _numeric_check(feature: str, ref: dict, window_df: pd.DataFrame) -> dict:
    live_props = bin_numeric(window_df[feature], ref["bin_edges"])
    psi = compute_psi(ref["reference_proportions"], live_props)
    return {"name": feature, "type": "numeric", "psi": psi, "tier": severity_tier(psi)}


def _categorical_check(feature: str, ref: dict, window_df: pd.DataFrame) -> dict:
    known = ref["known_categories"]
    live_props_dict = category_proportions(window_df[feature], known)
    ref_props = [ref["reference_proportions"].get(c, 0.0) for c in known] + [
        ref["reference_proportions"].get("__unexpected__", 0.0)
    ]
    live_props = [live_props_dict[c] for c in known] + [live_props_dict["__unexpected__"]]
    psi = compute_psi(ref_props, live_props)
    return {
        "name": feature,
        "type": "categorical",
        "psi": psi,
        "tier": severity_tier(psi),
        "unexpected_category_rate": unexpected_category_rate(window_df[feature], known),
    }


def _prediction_check(ref: dict, window_df: pd.DataFrame) -> dict:
    live_props = bin_numeric(window_df["predicted_score"], ref["bin_edges"])
    psi = compute_psi(ref["reference_proportions"], live_props)
    return {
        "name": "predicted_score",
        "type": "prediction",
        "psi": psi,
        "tier": severity_tier(psi),
        "reference_mean": ref["reference_mean"],
        "live_mean": float(window_df["predicted_score"].mean()),
    }


def build_drift_report(reference: dict, window_df: pd.DataFrame) -> dict:
    checks = []
    for feature, ref in reference["numeric_features"].items():
        checks.append(_numeric_check(feature, ref, window_df))
    for feature, ref in reference["categorical_features"].items():
        checks.append(_categorical_check(feature, ref, window_df))
    checks.append(_prediction_check(reference["prediction"], window_df))

    worst = max(checks, key=lambda c: c["psi"])
    tiers = [c["tier"] for c in checks]
    if "significant" in tiers:
        verdict = "significant"
    elif "moderate" in tiers:
        verdict = "moderate"
    else:
        verdict = "stable"

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "window_size_used": len(window_df),
        "verdict": verdict,
        "worst_check": {"name": worst["name"], "psi": worst["psi"], "tier": worst["tier"]},
        "checks": checks,
    }