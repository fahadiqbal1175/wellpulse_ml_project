"""Phase 6 verification: a logged run carries the right tags,
registration + Staging alias works, and the promotion decision picks
correctly in both the "first model vs. mean baseline" branch and the
"challenger vs. existing Production" branch."""
from __future__ import annotations

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor

from src.experiments import mlflow_utils


@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    """Points mlflow_utils at a throwaway sqlite db + artifact dir for
    this test only, so tests never touch (or collide with) the real
    project mlflow.db / mlruns/."""
    db_path = tmp_path / "test_mlflow.db"
    artifact_root = tmp_path / "mlruns"
    monkeypatch.setattr(mlflow_utils, "TRACKING_URI", f"sqlite:///{db_path}")
    monkeypatch.setattr(mlflow_utils, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(mlflow_utils, "EXPERIMENT_NAME", "test_experiment")
    monkeypatch.setattr(mlflow_utils, "_initialized", False)
    yield
    if mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def toy_xy():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    return X, y


def test_log_model_run_carries_standard_tags_and_metrics(isolated_mlflow, toy_xy) -> None:
    X, y = toy_xy
    model = DecisionTreeRegressor(random_state=42).fit(X, y)

    tags = mlflow_utils.standard_tags(phase="test_phase", stage="test_stage", model_type="DecisionTree")
    run_id = mlflow_utils.log_model_run(
        run_name="test_run",
        model=model,
        params=model.get_params(),
        metrics={"test_MAE": 0.1, "test_RMSE": 0.2, "test_R2": 0.95},
        tags=tags,
    )

    client = MlflowClient()
    run = client.get_run(run_id)
    assert run.data.tags["phase"] == "test_phase"
    assert run.data.tags["stage"] == "test_stage"
    assert run.data.tags["dataset_version"] == mlflow_utils.get_dataset_version()
    assert run.data.tags["feature_version"] == mlflow_utils.FEATURE_VERSION
    assert run.data.metrics["test_MAE"] == pytest.approx(0.1)
    assert run.data.metrics["test_RMSE"] == pytest.approx(0.2)


def test_register_first_model_promotes_when_it_clears_the_margin(isolated_mlflow, toy_xy) -> None:
    X, y = toy_xy
    model = DecisionTreeRegressor(random_state=42).fit(X, y)
    run_id = mlflow_utils.log_model_run(
        run_name="first_model_run",
        model=model,
        params=model.get_params(),
        metrics={"test_MAE": 0.5},
        tags=mlflow_utils.standard_tags(phase="test", stage="final_model"),
    )

    decision = mlflow_utils.register_and_maybe_promote(
        run_id=run_id,
        model_name="test_registered_model_a",
        test_mae=0.5,
        mean_baseline_test_mae=1.0,
        meaningful_margin_pct=20.0,
    )

    assert decision["criterion"] == "beat_mean_baseline_by_margin"
    assert decision["current_production_version"] is None
    assert decision["promoted_to_production"] is True

    client = MlflowClient()
    prod = client.get_model_version_by_alias("test_registered_model_a", "Production")
    assert prod.version == decision["new_version"]


def test_register_first_model_stays_staging_when_it_misses_the_margin(isolated_mlflow, toy_xy) -> None:
    X, y = toy_xy
    model = DecisionTreeRegressor(random_state=42).fit(X, y)
    run_id = mlflow_utils.log_model_run(
        run_name="weak_first_model_run",
        model=model,
        params=model.get_params(),
        metrics={"test_MAE": 0.95},
        tags=mlflow_utils.standard_tags(phase="test", stage="final_model"),
    )

    decision = mlflow_utils.register_and_maybe_promote(
        run_id=run_id,
        model_name="test_registered_model_b",
        test_mae=0.95,
        mean_baseline_test_mae=1.0,
        meaningful_margin_pct=20.0,
    )

    assert decision["promoted_to_production"] is False
    assert decision["staging_alias_set"] is True

    client = MlflowClient()
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias("test_registered_model_b", "Production")
    staging = client.get_model_version_by_alias("test_registered_model_b", "Staging")
    assert staging.version == decision["new_version"]


def test_register_challenger_vs_existing_production(isolated_mlflow, toy_xy) -> None:
    X, y = toy_xy
    model_name = "test_registered_model_c"

    # v1 establishes Production.
    model1 = DecisionTreeRegressor(random_state=42).fit(X, y)
    run_id_1 = mlflow_utils.log_model_run(
        run_name="v1_run",
        model=model1,
        params=model1.get_params(),
        metrics={"test_MAE": 0.5},
        tags=mlflow_utils.standard_tags(phase="test", stage="final_model"),
    )
    decision_1 = mlflow_utils.register_and_maybe_promote(
        run_id=run_id_1, model_name=model_name, test_mae=0.5,
        mean_baseline_test_mae=1.0, meaningful_margin_pct=20.0,
    )
    assert decision_1["promoted_to_production"] is True

    # v2 is WORSE than v1's Production MAE -> must not promote.
    model2 = DummyRegressor(strategy="mean").fit(X, y)
    run_id_2 = mlflow_utils.log_model_run(
        run_name="v2_run",
        model=model2,
        params=model2.get_params(),
        metrics={"test_MAE": 0.8},
        tags=mlflow_utils.standard_tags(phase="test", stage="final_model"),
    )
    decision_2 = mlflow_utils.register_and_maybe_promote(
        run_id=run_id_2, model_name=model_name, test_mae=0.8,
        mean_baseline_test_mae=1.0, meaningful_margin_pct=20.0,
    )
    assert decision_2["criterion"] == "beat_current_production_test_mae"
    assert decision_2["current_production_version"] == decision_1["new_version"]
    assert decision_2["promoted_to_production"] is False

    # v3 is BETTER than v1's Production MAE -> must promote, moving the alias.
    model3 = DecisionTreeRegressor(max_depth=1, random_state=7).fit(X, y)
    run_id_3 = mlflow_utils.log_model_run(
        run_name="v3_run",
        model=model3,
        params=model3.get_params(),
        metrics={"test_MAE": 0.2},
        tags=mlflow_utils.standard_tags(phase="test", stage="final_model"),
    )
    decision_3 = mlflow_utils.register_and_maybe_promote(
        run_id=run_id_3, model_name=model_name, test_mae=0.2,
        mean_baseline_test_mae=1.0, meaningful_margin_pct=20.0,
    )
    assert decision_3["promoted_to_production"] is True

    client = MlflowClient()
    prod = client.get_model_version_by_alias(model_name, "Production")
    assert prod.version == decision_3["new_version"]
