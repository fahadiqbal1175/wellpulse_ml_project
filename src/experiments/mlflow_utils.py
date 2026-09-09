"""
Phase 6 — MLflow & Model Registry (Section 16/17): shared helpers so
every training script (baseline.py, advanced.py, tuning.py's logger,
register_final_model.py) talks to MLflow the same way — one tracking
URI, one experiment, one tagging convention.

Tracking store: SQLite (`mlflow.db` at the project root) rather than a
plain local file store, because the Model Registry (register_model,
alias get/set) requires a database-backed store — a file store raises
at registration time. Artifacts still live on the local filesystem
under `mlruns/` (already gitignored), pointed to explicitly via each
experiment's `artifact_location` so `mlflow ui` resolves them
correctly regardless of the working directory it's launched from.

Promotion uses MLflow's alias mechanism (`set_registered_model_alias`
/ `get_model_version_by_alias`), not the deprecated stage-transition
API (`transition_model_version_stage`), per the blueprint's own
wording ("the Production alias moved").
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "wellpulse_mental_health_score"
REGISTERED_MODEL_NAME = "wellpulse_final_model"
DATASET_VERSION_FILE = PROJECT_ROOT / "data" / "raw" / "dataset_version.txt"

# Bumped by hand if src/features/*.py's transform logic ever changes.
# Unchanged since Phase 2 (add_engineered_features + CategoricalFeatureEncoder).
FEATURE_VERSION = "v1-phase2"

_initialized = False


def init_mlflow() -> str:
    """
    Idempotent: sets the tracking URI once, creates the
    `wellpulse_mental_health_score` experiment with an explicit
    artifact_location on first call, and simply reuses it on every
    later call (from this process or a fresh one, since the sqlite
    file persists on disk).

    Returns the experiment_id.
    """
    global _initialized
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        artifact_location = (ARTIFACT_ROOT / EXPERIMENT_NAME).resolve().as_uri()
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME, artifact_location=artifact_location
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)
    _initialized = True
    return experiment_id


def get_dataset_version() -> str:
    """Parses the `dataset_version=...` line from
    data/raw/dataset_version.txt (written by `src.data.ingest`).
    Returns "unknown" if the file or line is missing rather than
    raising — a missing version tag shouldn't block a training run."""
    if not DATASET_VERSION_FILE.exists():
        return "unknown"
    for line in DATASET_VERSION_FILE.read_text().splitlines():
        if line.startswith("dataset_version="):
            return line.split("=", 1)[1].strip()
    return "unknown"


def standard_tags(phase: str, stage: str, **extra: Any) -> dict[str, str]:
    """Every MLflow run in this project carries at least phase, stage,
    dataset_version, and feature_version — the four fields the
    blueprint's Section 16 diagram lists under an Experiment run.
    Values are cast to str since MLflow tags must be strings."""
    tags = {
        "phase": phase,
        "stage": stage,
        "dataset_version": get_dataset_version(),
        "feature_version": FEATURE_VERSION,
    }
    tags.update(extra)
    return {k: str(v) for k, v in tags.items()}


def _clean_params(params: dict[str, Any]) -> dict[str, str]:
    """MLflow log_params wants primitives; a couple of sklearn/
    LightGBM/XGBoost hyperparameters can be None or numpy scalars,
    which log_params tolerates, but a stray nested object (rare, but
    e.g. a callable) would not — stringify everything defensively."""
    return {str(k): str(v) for k, v in params.items()}


def _clean_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Only numeric values are valid MLflow metrics — non-numeric
    entries (there shouldn't be any in this project, but defensively)
    are dropped rather than raising."""
    out = {}
    for k, v in metrics.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def log_model_run(
    run_name: str,
    model,
    params: dict[str, Any],
    metrics: dict[str, Any],
    tags: dict[str, Any],
    extra_artifacts: list[str] | None = None,
    input_example=None,
) -> str:
    """
    Opens one MLflow run, logs params/metrics/tags, logs `model` under
    the artifact path "model" (so it can be registered later via
    `runs:/<run_id>/model`), and attaches any extra local file paths
    (plots, CSVs, the full joblib bundle) as artifacts.

    Returns the run_id.
    """
    init_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(_clean_params(params))
        mlflow.log_metrics(_clean_metrics(metrics))
        with warnings.catch_warnings():
            # Cosmetic only: MLflow warns that an integer column in the
            # inferred input schema can't represent NaN at inference
            # time. None of this project's features are ever null at
            # inference (Phase 1's schema forbids it), so this is safe
            # to silence rather than let it clutter every training run's
            # console output.
            warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.*")
            # serialization_format="cloudpickle", not MLflow 3.x's new
            # default "skops": skops refuses to deserialize a handful of
            # "untrusted" third-party types unless explicitly whitelisted,
            # and LightGBM/XGBoost's own Booster/regressor classes are on
            # that untrusted list — logging would raise for every model in
            # Phase 4's 9-model leaderboard except the plain sklearn ones.
            # cloudpickle handles all of Decision Tree / Random Forest /
            # Linear-family / LightGBM / XGBoost uniformly.
            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=input_example,
                serialization_format="cloudpickle",
            )
        for path in extra_artifacts or []:
            mlflow.log_artifact(str(path))
        return run.info.run_id


def register_and_maybe_promote(
    run_id: str,
    model_name: str,
    test_mae: float,
    mean_baseline_test_mae: float,
    meaningful_margin_pct: float = 20.0,
) -> dict[str, Any]:
    """
    Registers the model logged at `runs:/<run_id>/model` as a new
    version of `model_name`, sets the "Staging" alias on it, then
    decides on promotion per the blueprint's Section 17 criteria:

      - If a "Production" alias already exists for this registered
        model: promote only if `test_mae` beats that version's logged
        `test_MAE` metric.
      - Otherwise (first model ever registered under this name):
        promote only if `test_mae` beats `mean_baseline_test_mae` by
        at least `meaningful_margin_pct` percent.

    Promoting means moving the "Production" alias to the new version
    — MLflow aliases are unique per (model_name, alias), so setting it
    on the new version automatically un-points it from wherever it
    was before. Returns a decision record dict.
    """
    init_mlflow()
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri, model_name)
    client.set_registered_model_alias(model_name, "Staging", model_version.version)

    decision: dict[str, Any] = {
        "registered_model_name": model_name,
        "new_version": model_version.version,
        "run_id": run_id,
        "new_test_mae": test_mae,
        "mean_baseline_test_mae": mean_baseline_test_mae,
        "meaningful_margin_pct": meaningful_margin_pct,
        "staging_alias_set": True,
    }

    try:
        current_prod = client.get_model_version_by_alias(model_name, "Production")
    except MlflowException:
        current_prod = None

    if current_prod is not None:
        prod_run = client.get_run(current_prod.run_id)
        prod_test_mae = prod_run.data.metrics.get("test_MAE")
        decision["current_production_version"] = current_prod.version
        decision["current_production_test_mae"] = prod_test_mae
        should_promote = prod_test_mae is not None and test_mae < prod_test_mae
        decision["criterion"] = "beat_current_production_test_mae"
    else:
        threshold = mean_baseline_test_mae * (1 - meaningful_margin_pct / 100)
        decision["current_production_version"] = None
        decision["promotion_threshold_mae"] = threshold
        should_promote = test_mae <= threshold
        decision["criterion"] = "beat_mean_baseline_by_margin"

    decision["promoted_to_production"] = should_promote
    if should_promote:
        client.set_registered_model_alias(model_name, "Production", model_version.version)
        decision["reason"] = "Promotion criterion met — Production alias moved to new version."
    else:
        decision["reason"] = "Promotion criterion not met — new version stays at Staging only."

    return decision
