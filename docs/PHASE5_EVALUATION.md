# Phase 5 — Final Evaluation & Explainability

## Objective

Per `docs/PHASE4_TUNING.md`'s explicit next step: Phase 4 ended with a
FAIL verdict and no clear single winner (untuned Decision Tree led val
MAE; tuned Random Forest led val RMSE/R2 and was the more CV-robust
model). Phase 5 evaluates **both** candidates against the test fold —
the first and only time it is touched in this project — and lets that
decide the final model, rather than assuming tuning "should" have won.

## Step 1 — Final evaluation (`make evaluate`)

`src/evaluation/final_evaluation.py` reproduces the exact Phase 3/4
train/val/test split (`random_state=42`), refits the untuned Decision
Tree fresh (it was never saved as a file), loads the tuned Random
Forest from `models/phase4_best_model.joblib`, and scores both on
`split.test` for the first time anywhere in this project.

### Leaderboard (test set, `reports/phase5_final_evaluation.csv`)

| model | MAE | RMSE | R2 |
|---|---|---|---|
| Random Forest (tuned) | 0.2356 | 0.4613 | 0.8293 |
| Decision Tree (untuned) | 0.2358 | 0.5047 | 0.7956 |

**The tension resolves — barely, and in the tuned model's favor.**
MAE is now a statistical tie (0.2356 vs 0.2358, a 0.0002 gap), but
Random Forest clearly wins RMSE (0.461 vs 0.505) and R2 (0.829 vs
0.796). That matches `PHASE4_TUNING.md`'s own hypothesis: the untuned
tree's Phase 3/4 val MAE of 0.1509 was likely "a favorable-split
artifact" rather than a sign it truly generalizes best — on test, its
MAE nearly doubles (0.151 -> 0.236) while the tuned Random Forest's
MAE barely moves (0.170 -> 0.236, and it was never optimized for this
specific val fold the way the unconstrained tree implicitly was).
**Final model: Random Forest (tuned)**, selected by the blueprint's
stated primary metric (MAE), with RMSE/R2 both agreeing this time
instead of pulling the other way.

### Risk-tier classification report (`reports/phase5_risk_tier_report.csv`)

| risk_tier | precision | recall | f1 | support |
|---|---|---|---|---|
| low_risk | 0.500 | 1.000 | 0.667 | 16 |
| medium_risk | 0.746 | 0.733 | 0.739 | 60 |
| high_risk | 1.000 | 0.500 | 0.667 | 30 |

**A real limitation worth flagging for Section 31 (ethics):** the
model never *falsely* flags someone as high-risk (precision 1.0), but
it only *catches* half of the true high-risk students (recall 0.5) —
the other half get predicted into medium_risk. For a wellbeing
screening signal this is the direction of error that matters most: a
missed high-risk case is worse than an over-cautious one. This isn't
tuned away here — it's documented as a known limitation of a
regression-then-threshold approach on a ~700-row dataset, not
something a bigger model would necessarily fix.

## Step 2 — Error analysis (`make error-analysis`)

`src/evaluation/error_analysis.py` runs against the exact same final
model and test predictions — no separate fitting happens here.

**Predicted vs. actual** and **residuals vs. 4 major features**
(`Avg_Daily_Usage_Hours`, `Sleep_Hours_Per_Night`,
`Conflicts_Over_Social_Media`, `usage_to_sleep_ratio`) are saved to
`reports/figures/`. No systematic funnel or curve is visible in the
residual-vs-feature plots — errors look evenly scattered around zero
across the range of each feature, i.e. no obvious sign the model is
biased at high or low usage/sleep/conflict levels specifically.

### Subgroup performance

By `Academic_Level` (`reports/phase5_subgroup_by_academic_level.csv`):
Graduate (n=54, MAE 0.257) and Undergraduate (n=50, MAE 0.202) are
both reasonably estimated; High School (n=2) is too small to draw any
conclusion from (flagged via `reliable_n=False`).

By `Country` (`reports/phase5_subgroup_by_country.csv`, 40 countries
appear in test): most countries with 5+ test rows score well (India
MAE 0.012, South Korea 0.045, France 0.050), but **New Zealand stands
out badly** — MAE 1.249 across its 4 test rows, roughly 5x the overall
test MAE.

**Root cause, confirmed by inspection:** New Zealand has only 8 rows
in the entire 705-row dataset (3 train / 1 val / 4 test).
`RareCategoryBucketer` (`src/features/encoding.py`) keeps only the
top-8 most frequent countries and buckets everything else — including
New Zealand — into `Other`. The model literally cannot distinguish a
New Zealand respondent from an Ecuadorian or a Kenyan one; it's
predicting from the pooled `Other` pattern, not anything
New-Zealand-specific. This is a direct, visible consequence of the
Phase 2 bucketing design decision (documented at the time as a
deliberate overfitting-prevention tradeoff) — not a bug, but a real,
now-demonstrated cost of that tradeoff worth naming explicitly rather
than leaving implicit.

### Worst 10 predictions (`reports/phase5_worst_predictions.csv`)

3 of the 10 worst rows are the New Zealand cases above (all
under-predicted by 1.3-1.8 points). The rest are scattered
single-country misses (Czech Republic, Canada, Germany, Italy,
Maldives, USA) with no other obvious shared pattern — consistent with
"small-country bucketing" being the one clear, systematic failure
mode found here, rather than a broader feature-level blind spot.

## Step 3 — Explainability (`make explain`)

`src/evaluation/explainability.py` uses `shap.TreeExplainer` (exact,
since both candidates are tree-based) against the final model.

**SHAP summary plot** (`reports/figures/phase5_shap_summary.png`):
`usage_conflict_interaction` and `Conflicts_Over_Social_Media` are by
far the two most influential features — high values of both pull the
predicted score down sharply, low values pull it up. Everything else
(sleep, usage-to-sleep ratio, platform, country, demographics) has a
visibly smaller and more mixed effect. This matches the domain
intuition the blueprint's Section 9 feature design was built around:
usage alone matters less than usage *combined with* social friction.

**3 sample explanations** (one per risk tier, `reports/phase5_sample_explanations.json`),
each with a prediction, a 68% confidence interval (prediction +/- 1
residual std, measured on the val set so test's headline numbers
aren't spent twice), the top-3 SHAP factors, and a templated sentence,
e.g.:

> Test row 127 — actual=8.0, predicted=7.01, 68% interval=[6.70, 7.32]
> "This prediction was most shaped by usage-conflict interaction
> (raising the score), conflicts over social media (raising the
> score), and usage-to-sleep ratio (lowering the score)."

**Limitations (Section 14, restated here deliberately):** SHAP
explains *this model's* reasoning, not psychological causation; a
feature being a top contributor doesn't mean it caused the outcome;
and these explanations would shift if the model were retrained on new
data. The confidence interval above is a simple residual-spread proxy,
not a statistically calibrated prediction interval — it should be
described to end users as "typical variation," not a formal
confidence guarantee.

## Known limitations carried into Phase 6+

- The country-holdout generalization gap found in Phase 3
  (`reports/phase3_country_generalization.csv`: unseen-country MAE
  roughly double seen-country MAE) was not re-tested against the
  final tuned Random Forest here — Phase 5's test evaluation uses the
  *primary* stratified split, not the separate country-holdout split.
  Worth re-running before making any generalization claim in a demo
  or writeup.
- The New Zealand finding above is really a re-statement of that same
  gap at the level of one specific bucketed country, now with a
  concrete mechanism (rare-category bucketing) rather than an
  aggregate statistic.
- high_risk recall (0.5) is the single most important number to
  caveat if this project is ever framed as more than a portfolio/
  interview piece — it means half of true high-risk cases in this
  test set would not have been flagged as such.

## Reproducing these results

```bash
make evaluate        # writes reports/phase5_final_evaluation.csv,
                      # reports/phase5_risk_tier_report.csv,
                      # models/final_model.joblib
make error-analysis   # writes reports/figures/*.png,
                      # reports/phase5_subgroup_by_*.csv,
                      # reports/phase5_worst_predictions.csv
make explain          # writes reports/figures/phase5_shap_summary.png,
                      # reports/phase5_sample_explanations.json

# or all three in order:
make phase5
```

`make evaluate` must run at least once before `make error-analysis` or
`make explain`, since `models/final_model.joblib` (used by later
phases, e.g. Phase 6's registry and Phase 18's inference service) is
written there. In practice all three simply call
`build_final_eval_bundle()` fresh each time, so running them in any
order still produces correct (if redundant) results — `make phase5`
just avoids the redundancy.

## Next: Phase 6 — MLflow & Model Registry

Wrap training in MLflow runs, register the final model
(`models/final_model.joblib`'s bundled Random Forest + encoder), and
promote it to Production. Verify: `mlflow ui` shows the full run
history and the registry shows a Production version.
