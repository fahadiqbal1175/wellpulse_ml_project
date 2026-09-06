"""
Builds notebooks/01_eda.ipynb from the cell definitions below, then
executes it top-to-bottom so the committed notebook always has real,
reproduced outputs (Milestone ML-2: "EDA notebook runs top-to-bottom
and reproduces its own findings") rather than stale/hand-edited plots.

Run this whenever the EDA content changes:
    python3 scripts/build_eda_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_eda.ipynb"

# Each entry: ("markdown" | "code", source_string)
CELLS = [
("markdown", """\
# WellPulse v2 — Phase 2: EDA & Leakage Check

Dataset: Students' Social Media Addiction (Kaggle,
`adilshamim8/social-media-addiction-vs-relationships`), validated in
Phase 1 (`src/data/schema.py`). This notebook covers the EDA plan from
Section 8 of the blueprint and the leakage check the blueprint
explicitly calls for on `Addicted_Score` (Section 5) and
`Affects_Academic_Performance` (Section 4B)."""),

("code", """\
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.schema import validate_raw_dataset
from src.features.engineering import add_engineered_features
from src.features.encoding import CategoricalFeatureEncoder

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

df = pd.read_csv("../data/raw/students_social_media_addiction.csv")
df = validate_raw_dataset(df)
print(f"Loaded and validated: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()"""),

("markdown", "## 1. Missingness and duplicate check"),

("code", """\
print("Nulls per column:")
print(df.isnull().sum())
print()
print("Duplicate Student_ID rows:", df["Student_ID"].duplicated().sum())"""),

("markdown", """\
**Finding:** zero missing values, zero duplicate IDs — confirms the
Phase 0 claim and the Pandera schema's `not_nullable`/`unique`
constraints are consistent with the actual data."""),

("markdown", "## 2. Univariate distributions"),

("code", """\
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
sns.histplot(df["Mental_Health_Score"], bins=range(1, 12), ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Mental_Health_Score (target)")
sns.histplot(df["Avg_Daily_Usage_Hours"], bins=20, ax=axes[0, 1], color="#DD8452")
axes[0, 1].set_title("Avg_Daily_Usage_Hours")
sns.histplot(df["Sleep_Hours_Per_Night"], bins=20, ax=axes[1, 0], color="#55A868")
axes[1, 0].set_title("Sleep_Hours_Per_Night")
sns.histplot(df["Addicted_Score"], bins=range(1, 12), ax=axes[1, 1], color="#C44E52")
axes[1, 1].set_title("Addicted_Score")
plt.tight_layout()
plt.show()"""),

("markdown", """\
**Finding:** `Mental_Health_Score` and `Addicted_Score` both look
close to uniformly spread across their range rather than showing the
skew or clustering typical of real self-report survey scales — an
early hint that will matter for the leakage check below. No transform
(e.g. log) looks necessary for the target given this shape."""),

("markdown", "## 3. Bivariate analysis, correlation, and the leakage check"),

("code", """\
numeric_cols = [
    "Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night",
    "Conflicts_Over_Social_Media", "Addicted_Score", "Mental_Health_Score",
]
corr = df[numeric_cols].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Correlation matrix (numeric features + target)")
plt.tight_layout()
plt.show()

corr["Mental_Health_Score"].drop("Mental_Health_Score").sort_values()"""),

("markdown", """\
**Finding:** every numeric feature correlates strongly with the
target — `Addicted_Score` at **r = -0.95**, `Conflicts_Over_Social_Media`
at -0.89, `Avg_Daily_Usage_Hours` at -0.80. Real self-reported survey
data essentially never produces correlations this clean; this is
investigated quantitatively below rather than assumed."""),

("code", """\
def r_squared(x, y):
    coeffs = np.polyfit(x, y, 1)
    pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot

y = df["Mental_Health_Score"].values

r2_addicted_only = r_squared(df["Addicted_Score"].values, y)
r2_affects_only = r_squared(
    (df["Affects_Academic_Performance"] == "Yes").astype(int).values, y
)

X_all_numeric = df[
    ["Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Conflicts_Over_Social_Media"]
].values
X_all_numeric = np.column_stack([X_all_numeric, np.ones(len(X_all_numeric))])
coef, *_ = np.linalg.lstsq(X_all_numeric, y, rcond=None)
pred_all = X_all_numeric @ coef
r2_without_addicted = 1 - np.sum((y - pred_all) ** 2) / np.sum((y - y.mean()) ** 2)

print(f"R^2 using ONLY Addicted_Score:                          {r2_addicted_only:.3f}")
print(f"R^2 using ONLY Affects_Academic_Performance:            {r2_affects_only:.3f}")
print(f"R^2 using Age+Usage+Sleep+Conflicts (no Addicted_Score): {r2_without_addicted:.3f}")"""),

("markdown", """\
**Leakage check verdict (see `docs/FEATURES.md` for the full decision
record):**

- `Addicted_Score` alone explains 89.3% of the target's variance with
  a plain linear fit. Combined with every other numeric feature it
  only adds ~7 points of R² on top of what the *other* features
  already provide (82.3% → 89.6%) — meaning `Addicted_Score` is doing
  the work of a near-direct restatement of the target, not
  contributing independent behavioral signal. **Decision: excluded**
  from the model's feature set.
- `Affects_Academic_Performance` alone explains 65.4% of variance —
  consistent with the blueprint's own suspicion (Section 4B) that
  it's a self-assessment entangled with overall wellbeing rather than
  an independent measurement. It was already outside Section 5's
  input list; this confirms that exclusion was the right call.
- Even *without* `Addicted_Score`, the remaining behavioral features
  reach R² = 0.823 — still unusually clean for organic self-report
  data. This is documented as a dataset-quality caveat in
  `docs/FEATURES.md`, not hidden: the methodology (validation, honest
  CV, held-out test set) is unaffected either way, but the *strength*
  of the eventual model's performance shouldn't be over-interpreted
  as proof of a strong real-world causal relationship."""),

("markdown", "## 4. Outlier check (flag, don't drop)"),

("code", """\
print("Max Avg_Daily_Usage_Hours:", df["Avg_Daily_Usage_Hours"].max())
print("Min Sleep_Hours_Per_Night:", df["Sleep_Hours_Per_Night"].min())
print("Rows with usage > 10 hrs/day:", (df["Avg_Daily_Usage_Hours"] > 10).sum())
print("Rows with sleep < 4 hrs/night:", (df["Sleep_Hours_Per_Night"] < 4).sum())"""),

("markdown", """\
**Finding:** no implausible values present (max usage 8.5 hrs/day, min
sleep 3.8 hrs/night — both physically plausible for a student
population). Nothing to flag or drop."""),

("markdown", "## 5. Class balance check for the display-only risk tier"),

("code", """\
print(df["Mental_Health_Score"].value_counts().sort_index())
print()
print(df["Mental_Health_Score"].quantile([0.25, 0.5, 0.75]))"""),

("markdown", """\
**First attempt, rejected:** an initial `score <= 4` / `5-7` / `>= 8`
split put only 4.1% of rows in the high-risk tier (score=4 is the
observed minimum, and score=9 has a single row) — the UI would make
high-risk look far rarer than the underlying score distribution
actually supports. Rejecting a boundary because it's *convenient*
rather than *representative* is the same discipline the blueprint
applies to the dataset/target choice itself (Section 4)."""),

("code", """\
from src.features.engineering import compute_risk_tier

tier_counts = compute_risk_tier(df).value_counts()
print(tier_counts)
print((tier_counts / len(df) * 100).round(1).astype(str) + "%")
tier_counts.plot(kind="bar", color=["#C44E52", "#DD8452", "#55A868"])
plt.title("Risk tier balance (display-only transform, not a trained classifier)")
plt.ylabel("count")
plt.show()"""),

("markdown", """\
**Finding:** using the 25th/75th percentiles as boundaries instead
(`<=5` / `6-7` / `>=8`) gives 28.7% high-risk, 56.3% medium-risk, 15.0%
low-risk — every tier has meaningful representation, so the UI won't
imply a rare tier is common. `compute_risk_tier()` (in
`src/features/engineering.py`) is the single source of truth for this
logic — it's reused for stratifying the train/val/test split in Phase 3
(`src/data/split.py`), so the notebook and the splitting code can never
drift out of sync."""),

("markdown", """\
## 6. Temporal analysis

**Not applicable.** As established in Phase 0/1 and Section 4 of the
blueprint, this is a single cross-sectional snapshot per respondent —
there is no time dimension to analyze. Noted explicitly here rather
than silently skipped, per the EDA plan."""),

("markdown", "## 7. Feature engineering — applying the finalized feature list"),

("code", """\
engineered = add_engineered_features(df)
model_ready = engineered.drop(columns=["Addicted_Score", "Affects_Academic_Performance", "Student_ID"])

encoder = CategoricalFeatureEncoder()
final = encoder.fit_transform(model_ready)

print(f"Raw: {df.shape} -> engineered: {engineered.shape} -> model-ready (encoded): {final.shape}")
final.head()"""),

("markdown", """\
## 8. Summary

- **Leakage check:** `Addicted_Score` excluded (near-tautological with
  the target); `Affects_Academic_Performance` confirmed excluded (as
  already specified in Section 5).
- **Dataset-quality caveat:** relationships in this dataset are
  cleaner than real self-report survey data typically is — documented
  as a limitation, not hidden.
- **Finalized feature list:** see `docs/FEATURES.md` for the full
  table and reasoning. `feature_version = "v1_phase2"`.
- **No temporal structure, no implausible outliers, balanced risk
  tiers, zero missing data** — dataset is clean and ready for Phase 3
  (baseline models)."""),
]


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(src) if kind == "markdown"
        else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    return nb


def main() -> None:
    nb = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    client.execute(cwd=str(NOTEBOOK_PATH.parent))

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Executed and wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
