"""
Phase 7 — Recommendation lookup (Section 15/18).

Not part of the blueprint's narrow Phase 7 milestone text (ML-6 only
requires score + SHAP factors), but Section 18's response pipeline
diagram places "Recommendation lookup" right after the SHAP step and
before the response is built, and Section 15 designs this module
specifically so it's cheap and self-contained to add: "rule-based,
not ML-based... a curated lookup table mapping each possible
top-SHAP-factor to 1-3 vetted suggestions... never touches the
trained model or its raw outputs beyond the ranked factor list."
Flagged here (and in the implementation guide) as an addition beyond
the narrow milestone rather than folded in silently — drop the
`get_recommendations` call in `inference.py` if you'd rather defer
this to a later phase.

Only factors "lowering" the predicted score get a recommendation —
factors "raising" it are positive contributors with nothing to act
on. Only the 5 continuous/engineered features get concrete,
actionable text; one-hot demographic factors (Gender, Country,
Academic_Level, Relationship_Status, age_group, Most_Used_Platform)
get a neutral contextual fallback rather than invented behavior-
change advice tied to a trait that isn't actionable.
"""
from __future__ import annotations

ACTIONABLE_RECOMMENDATIONS: dict[str, str] = {
    "Avg_Daily_Usage_Hours": (
        "Consider setting a daily screen-time limit or scheduling deliberate "
        "offline blocks, especially around study and sleep hours."
    ),
    "Sleep_Hours_Per_Night": (
        "Prioritizing a consistent sleep schedule (aiming for 7-9 hours) is one "
        "of the most reliable levers for wellbeing."
    ),
    "Conflicts_Over_Social_Media": (
        "Frequent conflict over social media use is worth addressing directly — "
        "a candid conversation with the people involved, or adjusting the "
        "notification/sharing settings that tend to trigger friction, can help."
    ),
    "usage_to_sleep_ratio": (
        "Usage relative to sleep suggests screen time may be cutting into rest — "
        "try moving your last usage session earlier in the evening."
    ),
    "usage_conflict_interaction": (
        "High usage combined with social friction compounds risk — reducing "
        "either lever (usage or the underlying conflict) should help more than "
        "addressing just one alone."
    ),
}

_GENERIC_FALLBACK = (
    "{pretty_name_capitalized} is a contributing factor in this prediction. "
    "It reflects context rather than something to directly change, but may be "
    "worth discussing with a counselor or mentor."
)

MAX_RECOMMENDATIONS = 3


def get_recommendations(factors: list[dict]) -> list[str]:
    """`factors` is the exact list `top_n_factors()`
    (`src/evaluation/explainability.py`) returns — this function reads
    only `feature`, `pretty_name`, and `direction` from it."""
    recommendations: list[str] = []
    for factor in factors:
        if factor["direction"] != "lowering":
            continue
        text = ACTIONABLE_RECOMMENDATIONS.get(factor["feature"])
        if text is None:
            text = _GENERIC_FALLBACK.format(
                pretty_name_capitalized=factor["pretty_name"][:1].upper()
                + factor["pretty_name"][1:]
            )
        recommendations.append(text)
    return recommendations[:MAX_RECOMMENDATIONS]
