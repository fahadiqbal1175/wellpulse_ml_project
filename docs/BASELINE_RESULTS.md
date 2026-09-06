# Phase 3 — Baseline Model Results

Data splitting (Section 10): 70/15/15 train/val/test, stratified on
the risk tier (`src/data/split.py`, thresholds from `docs/FEATURES.md`
via `compute_risk_tier()`). **The test fold is not used anywhere in
this document** — every number below is on the validation set, kept
that way deliberately so test stays a single-use, unbiased estimate
for Phase 5's final evaluation.

## Leaderboard (val set, `reports/phase3_leaderboard.csv`)

| model | MAE | RMSE | R² |
|---|---|---|---|
| Decision Tree | 0.151 | 0.389 | 0.873 |
| Random Forest | 0.193 | 0.298 | 0.925 |
| Linear Regression | 0.310 | 0.436 | 0.840 |
| Ridge | 0.313 | 0.450 | 0.829 |
| Lasso | 0.384 | 0.529 | 0.764 |
| Dummy (median) | 0.858 | 1.112 | -0.040 |
| Dummy (mean) | 0.909 | 1.090 | -0.00003 |

7 model families trained and compared on the identical train/val
split, comfortably clearing Milestone ML-3's minimum of 4. Every
non-dummy model beats both dummy baselines by a wide margin (as it
should — if a model can't beat guessing the mean, it isn't learning
anything).

**A genuinely interesting result, not smoothed over:** Decision Tree
has the *lowest MAE* (the blueprint's stated primary metric, Section
13) but a *worse* RMSE and R² than Random Forest. That means Random
Forest's errors are more consistently small, while Decision Tree's are
smaller *on average* but with some larger misses that hurt it more
under a squared-error metric. Section 11 is explicit that "the final
model is whichever wins on the validation metric, not assumed in
advance" — by the stated primary metric (MAE), **Decision Tree is the
current Phase 3 leader**, though Random Forest is the safer choice if
occasional large errors matter more than average error. This tension
is exactly what Phase 4's tuning and Phase 5's full evaluation are for
— it isn't resolved here, just reported honestly.

## Country generalization check (`reports/phase3_country_generalization.csv`)

A Random Forest trained on 88 countries, evaluated on 22 entirely
unseen countries vs. an equally-sized held-back sample from the
*training* countries:

| eval_set | n_rows | MAE | RMSE | R² |
|---|---|---|---|---|
| seen_countries (held-back rows) | 190 | 0.161 | 0.257 | 0.953 |
| unseen_countries (fully held out) | 190 | 0.391 | 0.536 | 0.699 |

**Finding:** error roughly **doubles** (MAE 0.16 → 0.39) and R² drops
from 0.95 to 0.70 when the model sees a country it never trained on.
This is a real, measured generalization gap — not hypothetical — and
directly confirms the bias risk flagged in Section 4 ("self-selected
survey respondents, likely concentrated in specific universities/
countries — generalization claims must be scoped accordingly"). Any
claim about this model's accuracy should be scoped to "students from
countries similar to those in the training data," not stated as a
universal accuracy figure. This belongs in the limitations
documentation for Section 31 and is a strong, concrete interview
answer to "how do you know your model generalizes?"

## Reproducing these results

```bash
python3 -m src.models.baseline
```

Writes both CSVs to `reports/` and prints the same tables shown above.

## Next: Phase 4 — Advanced Modeling & Tuning

LightGBM/XGBoost added to the comparison, plus randomized search on
the top 1-2 candidates from this leaderboard (likely Decision Tree
and/or Random Forest, given the MAE/RMSE split above). Verify: the
tuned model beats the best untuned baseline (currently Decision Tree
at MAE 0.151).
