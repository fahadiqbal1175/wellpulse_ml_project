# Phase 4 — Advanced Modeling & Tuning

## Objective

Add LightGBM/XGBoost to the model comparison, then run randomized
search (not full grid search) on the top-2 candidates by validation
MAE. Verify whether the tuned result beats Phase 3's best untuned
baseline (Decision Tree, MAE 0.1509).

## Step 1 — Extended leaderboard (`make advanced`)

Same train/val split, same encoder, same evaluation loop as Phase 3 —
`src.models.baseline.run_baseline_leaderboard` was given an optional
`model_factories` argument so the 9-model leaderboard is one single
reproducible run rather than a read-old-CSV-and-append.

| model | MAE | RMSE | R2 |
|---|---|---|---|
| Decision Tree | 0.1509 | 0.3885 | 0.8730 |
| Random Forest | 0.1931 | 0.2977 | 0.9254 |
| XGBoost | 0.2277 | 0.3628 | 0.8893 |
| LightGBM | 0.2619 | 0.3534 | 0.8949 |
| Linear Regression | 0.3100 | 0.4361 | 0.8400 |
| Ridge | 0.3130 | 0.4504 | 0.8294 |
| Lasso | 0.3835 | 0.5291 | 0.7645 |
| Dummy (median) | 0.8585 | 1.1117 | -0.0396 |
| Dummy (mean) | 0.9090 | 1.0903 | -0.0000 |

**Neither LightGBM nor XGBoost beat Decision Tree or Random Forest on
this dataset.** This isn't surprising in hindsight: with only 493
training rows, gradient boosting has little room to exploit its usual
advantage over simpler tree ensembles, and Phase 2 already flagged
this dataset as unusually clean/low-noise for self-report data — the
conditions under which a single well-fit tree can be hard to beat.

Confirms Phase 3's own expectation: the top-2 candidates by MAE are
still **Decision Tree** and **Random Forest** — verified empirically
here rather than assumed, since LightGBM/XGBoost could plausibly have
displaced them.

## Step 2 — Randomized search (`make tune`)

- Search: `RandomizedSearchCV`, 60 iterations, 5-fold CV, scored on
  `neg_mean_absolute_error` (matches the blueprint's primary metric)
- CV folds drawn from **train only** (493 rows); val is scored once at
  the end per candidate, exactly like Phase 3's baseline comparison;
  test remains untouched for Phase 5
- Every one of the 60+60 = 120 CV trials, plus each candidate's
  best-of-search summary, is logged via
  `src.experiments.logger.ExperimentLogger` to
  `reports/experiments/phase4_tuning.csv` and per-run JSON files under
  `reports/experiments/phase4_tuning/` — this is the log format Phase
  6's MLflow work will read from / eventually replace with real
  MLflow calls, with no change needed at the call sites in
  `src/models/tuning.py`

### Results

| Model | Untuned val MAE (Phase 3/4) | Best CV MAE (train, 5-fold) | Tuned val MAE | Tuned val RMSE | Tuned val R² |
|---|---|---|---|---|---|
| Decision Tree | 0.1509 | 0.1715 | 0.2170 | 0.3889 | 0.8728 |
| Random Forest | 0.1931 | 0.1477 | 0.1704 | 0.3119 | 0.9181 |

## Verdict

**Baseline to beat: MAE 0.1509 (untuned Decision Tree, Phase 3)**
**Best tuned result: MAE 0.1704 (tuned Random Forest) — FAIL**

Tuning did not beat the untuned baseline. Reported as-is rather than
re-run with a friendlier search space or a different metric, per this
project's practice of documenting inconvenient results instead of
picking around them (see Phase 2/3's leakage and country-generalization
findings).

## Discussion — why tuning didn't win here

Two things are going on simultaneously, and it's worth separating them:

1. **Tuned Random Forest genuinely improved over untuned Random
   Forest** (val MAE 0.1931 → 0.1704, val R² 0.9254 → 0.9181 — R² dipped
   slightly, RMSE improved 0.2977 → 0.3119 is actually slightly worse;
   MAE and RMSE moved in different directions, a smaller version of
   the same MAE-vs-RMSE tension Phase 3 already documented). Tuning
   the ensemble's regularization did what it's supposed to do.
2. **The untuned Decision Tree's val MAE (0.1509) still isn't beaten.**
   An unconstrained `DecisionTreeRegressor` (no depth limit, no
   pruning) fully memorizes patterns in the 493 train rows; on most
   datasets that overfits and val MAE would be worse than a
   regularized model. Here it happens to score lowest MAE on this
   *specific* 106-row val fold. The CV-based search evaluates
   robustness across 5 folds of train (best CV MAE 0.1715 for the
   tuned tree) — a fairer estimate of true generalization — but that
   more honest estimate is worse than the untuned tree's lucky single
   val score.

In short: the untuned Decision Tree's 0.1509 is likely **partly a
favorable-split artifact**, not proof that an unconstrained tree
truly generalizes best. This is consistent with earlier phases'
running theme that this dataset is small and unusually clean, which
makes single-split comparisons (like Phase 3's val-only leaderboard)
more sensitive to which specific rows land in val.

## Known limitations

- Tuning was validated on the same in-distribution val fold used in
  Phase 3, not the country-holdout split. Phase 3's finding — Random
  Forest's error roughly doubling on unseen countries (MAE 0.16 →
  0.39) — was not re-tested here and likely still applies to the
  tuned model. Worth re-checking before treating any model here as
  final.
- 5-fold CV on ~493 train rows means each fold is small (~98-99 rows);
  CV MAE estimates carry meaningfully more variance than they would on
  a larger dataset — visible above in how much the tree's CV MAE
  (0.1715) differs from its single-val-fold MAE (0.1509 untuned /
  0.2170 tuned).
- Given the FAIL verdict, Phase 5's final test-set evaluation should
  run for **both** the untuned Decision Tree (Phase 3's leaderboard
  winner) and the tuned Random Forest (this phase's best CV-validated
  model), rather than assuming tuning produced the model to carry
  forward.

## Next

Phase 5 — Final Evaluation (first and only look at the test fold) —
evaluate both the untuned Decision Tree and the tuned Random Forest
against test, since Phase 4 didn't produce a clear single winner.
