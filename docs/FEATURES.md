# Feature List — Finalized (Phase 2)

This is the Phase 2 deliverable required by Section 37's roadmap: "a
documented leakage check and a finalized feature list." The leakage
check's numbers (correlations, R² probes) come from
`notebooks/01_eda.ipynb`, Section 3 — this file is the resulting
decision record.

## Target

`Mental_Health_Score` (int, 1-10 in the observed data) — regression target.

## Leakage check outcome: `Addicted_Score` is EXCLUDED

Section 5 of the blueprint listed `Addicted_Score` as an input "checked
for leakage — see Section 8." The check:

- `Addicted_Score` alone predicts `Mental_Health_Score` with **R² = 0.893**
  (simple linear fit, no other features).
- All other numeric features *combined* (Age, Usage, Sleep, Conflicts,
  without Addicted_Score) reach R² = 0.823 — still very high for
  self-report survey data, but that's a separate data-quality caveat
  (see below), not a leakage decision by itself.
- Every raw feature is also suspiciously strongly correlated with the
  target (Addicted_Score −0.95, Conflicts −0.89, Usage −0.80, Sleep
  +0.71) — correlations this clean are unusual for genuinely noisy
  self-reported survey data, and are consistent with `Addicted_Score`
  and `Mental_Health_Score` being two outputs of the *same* underlying
  generation process rather than independently-measured quantities.

**Decision:** `Addicted_Score` is excluded from the model's feature
set. Training on it would mean the model is mostly just inverting a
near-linear formula (`Mental_Health_Score ≈ 11 − 0.66 × Addicted_Score`)
rather than learning from genuinely independent behavioral signals —
not a meaningful modeling exercise, and not a defensible "SHAP explains
real behavioral drivers" story for Section 14's explainability work if
the top driver is essentially the same score restated.

`Affects_Academic_Performance` was already excluded from Section 5's
input list in the original blueprint (not just flagged as a risk) —
confirmed as the right call here too: it alone reaches R² = 0.654
against the target, consistent with the blueprint's own suspicion
(Section 4B) that it's answered with the respondent's overall wellbeing
already in mind, not measured independently of it.

## Dataset-quality caveat (beyond what the blueprint anticipated)

Even after removing `Addicted_Score`, the remaining behavioral features
explain ~82% of the target's variance with a *plain linear fit* — real
self-reported survey data is almost never this clean. This dataset
likely has a synthetic or template-generated component rather than
being pure organic survey noise. This doesn't block the project (the
blueprint's regression framing is still the right approach, and a
held-out test set will still measure real generalization), but it
belongs in the "limitations" documentation for Section 31 and is
worth having a one-sentence answer ready for an interviewer: *"the
relationships in this dataset are cleaner than real self-report data
usually is, which likely reflects some synthetic/templated element in
how the Kaggle dataset was constructed — the pipeline and methodology
are built to be identical either way, but I'm not overclaiming this as
noisy real-world signal."*

## Final feature set

| Feature | Type | Source | Notes |
|---|---|---|---|
| `Avg_Daily_Usage_Hours` | numeric | raw | |
| `Sleep_Hours_Per_Night` | numeric | raw | |
| `Conflicts_Over_Social_Media` | numeric | raw | |
| `Age` | numeric | raw | |
| `usage_to_sleep_ratio` | numeric | engineered | usage / sleep |
| `usage_conflict_interaction` | numeric | engineered | usage × conflicts |
| `Gender` | categorical (one-hot) | raw | 2 categories |
| `Academic_Level` | categorical (one-hot) | raw | 3 categories |
| `Most_Used_Platform` | categorical (one-hot) | raw | 12 categories |
| `Relationship_Status` | categorical (one-hot) | raw | 3 categories |
| `age_group` | categorical (one-hot) | engineered | 4 bins |
| `Country` | categorical (bucketed top-8 + Other, one-hot) | raw | 110 raw values → 9 |

**Excluded:** `Addicted_Score` (leakage, see above), `Affects_Academic_Performance`
(leakage, excluded per original blueprint spec), `Student_ID` (identifier,
no predictive meaning).

`feature_version = "v1_phase2"` — bump this string in MLflow (Phase 6)
any time this list changes, per Section 9's reproducibility requirement.
