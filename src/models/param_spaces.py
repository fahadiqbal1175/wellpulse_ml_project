"""
Randomized-search hyperparameter distributions for Phase 4 tuning
(`src/models/tuning.py`). Only model names with an entry here are
eligible to be selected as a tuning candidate — keys must match the
`model` column values produced by `src.models.advanced.run_phase4_leaderboard`.
"""
from __future__ import annotations

from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.models.baseline import RANDOM_STATE

# n_jobs=1 on every base estimator deliberately: RandomizedSearchCV
# already parallelizes across CV folds x param combos with n_jobs=-1;
# parallelizing inside each model too would oversubscribe cores and
# often runs *slower*, not faster.
TUNING_MODEL_FACTORIES = {
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
    "LightGBM": lambda: LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, n_jobs=1),
    "XGBoost": lambda: XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=1),
}

PARAM_SPACES = {
    "Decision Tree": {
        "max_depth": randint(2, 20),
        "min_samples_split": randint(2, 40),
        "min_samples_leaf": randint(1, 20),
        "max_features": [None, "sqrt", "log2", 0.5, 0.7, 0.9],
        "ccp_alpha": uniform(0.0, 0.02),
    },
    "Random Forest": {
        "n_estimators": randint(100, 800),
        "max_depth": randint(3, 30),
        "min_samples_split": randint(2, 40),
        "min_samples_leaf": randint(1, 20),
        "max_features": [None, "sqrt", "log2", 0.5, 0.7],
        "bootstrap": [True, False],
    },
    "LightGBM": {
        "n_estimators": randint(100, 800),
        "num_leaves": randint(7, 128),
        "max_depth": randint(-1, 16),
        "learning_rate": uniform(0.01, 0.29),
        "min_child_samples": randint(5, 60),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 1.0),
    },
    "XGBoost": {
        "n_estimators": randint(100, 800),
        "max_depth": randint(2, 12),
        "learning_rate": uniform(0.01, 0.29),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 1.0),
        "gamma": uniform(0.0, 0.5),
    },
}
