"""
Project-wide test isolation for MLflow (Phase 6).

Several tests exercise code paths that call into MLflow only as a
SIDE EFFECT of testing something else — most notably
`ExperimentLogger.log_run()`'s Phase 6 dual-write, exercised by
`tests/test_tuning.py`, which has nothing to do with MLflow itself
but would otherwise write real rows into this project's real
`mlflow.db` / `mlruns/` every time the test suite runs.

This autouse fixture redirects every test's MLflow writes to a fresh,
throwaway sqlite db + artifact dir under pytest's own `tmp_path`, so
running `pytest` can never pollute the tracking store that `mlflow
ui` and `python -m src.experiments.register_final_model` use for the
project's real run/registry history. `tests/test_mlflow_registry.py`
also sets up its own isolation explicitly (so it still passes even if
this fixture is ever removed) — the two layering is harmless since
both just point at a unique, empty `tmp_path` per test.
"""
import pytest

from src.experiments import mlflow_utils


@pytest.fixture(autouse=True)
def _isolate_mlflow_tracking_store(tmp_path, monkeypatch):
    db_path = tmp_path / "conftest_mlflow.db"
    artifact_root = tmp_path / "conftest_mlruns"
    monkeypatch.setattr(mlflow_utils, "TRACKING_URI", f"sqlite:///{db_path}")
    monkeypatch.setattr(mlflow_utils, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(mlflow_utils, "_initialized", False)
    yield
