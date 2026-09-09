"""
Lightweight experiment logger for Phase 4 hyperparameter tuning.

Mimics the MLflow API surface (log_run ~ start_run + log_params +
log_metrics) so that swapping this for a real MLflow-backed logger in
Phase 6 requires touching only this file, not any calling code in
`src/models/tuning.py`.

Phase 6 update: `log_run()` now DUAL-WRITES — the original CSV/JSON
output is unchanged (tests/test_tuning.py still asserts on
`logger.csv_path` row counts), and each call additionally opens one
MLflow run tagged with dataset_version/feature_version/phase/stage so
every one of tuning.py's ~120 CV trials + 2 best-of-search rows shows
up in `mlflow ui`, satisfying Milestone ML-4 ("all runs visible and
comparable"). Each MLflow run is flat (not nested) and tagged with
this logger's `experiment` name plus the run's own tags, so the full
set is filterable in the UI (e.g. `tags.experiment = "phase4_tuning"
AND tags.stage = "best_of_search"`) without needing a long-lived
parent run left open across the whole script's lifetime.

The MLflow write is best-effort: a failure there (e.g. a locked
sqlite file under concurrent access) is logged to stdout and swallowed
rather than raised, so it can never break the CSV/JSON path that
tuning.py's own tests depend on.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.experiments.mlflow_utils import init_mlflow, standard_tags


@dataclass
class ExperimentRun:
    run_id: str
    experiment: str
    model_name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    ended_at: str = ""

    def to_flat_dict(self) -> dict[str, Any]:
        flat = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "model_name": self.model_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }
        flat.update({f"param_{k}": v for k, v in self.params.items()})
        flat.update({f"metric_{k}": v for k, v in self.metrics.items()})
        flat.update({f"tag_{k}": v for k, v in self.tags.items()})
        return flat


class ExperimentLogger:
    """
    Append-only experiment tracker. Each `log_run()` call writes:
      - one row to <output_dir>/<experiment>.csv (flat, easy to
        `pd.read_csv` and sort)
      - one JSON file to <output_dir>/<experiment>/<run_id>.json
        (full param/metric/tag detail)

    Phase 6 swap plan: replace the body of `log_run()` with
    `mlflow.start_run()` / `mlflow.log_params()` / `mlflow.log_metrics()`
    — call sites in tuning.py don't change.
    """

    def __init__(self, experiment: str, output_dir: str | Path = "reports/experiments"):
        self.experiment = experiment
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detail_dir = self.output_dir / experiment
        self.detail_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / f"{experiment}.csv"

    def log_run(
        self,
        model_name: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        tags: dict[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> ExperimentRun:
        run = ExperimentRun(
            run_id=str(uuid.uuid4())[:8],
            experiment=self.experiment,
            model_name=model_name,
            params=params,
            metrics=metrics,
            tags=tags or {},
            started_at=(started_at or datetime.now(timezone.utc)).isoformat(),
            ended_at=datetime.now(timezone.utc).isoformat(),
        )
        self._append_csv_row(run.to_flat_dict())
        self._write_json(run)
        self._log_to_mlflow(run)
        return run

    def _log_to_mlflow(self, run: ExperimentRun) -> None:
        try:
            init_mlflow()
            mlflow_tags = standard_tags(
                phase=str(run.tags.get("phase", self.experiment)),
                stage=str(run.tags.get("stage", "unknown")),
                experiment=self.experiment,
                model_type=run.model_name,
                source_run_id=run.run_id,
            )
            with mlflow.start_run(run_name=f"{self.experiment}_{run.model_name}_{run.run_id}"):
                mlflow.set_tags(mlflow_tags)
                mlflow.log_params({str(k): str(v) for k, v in run.params.items()})
                numeric_metrics = {}
                for k, v in run.metrics.items():
                    try:
                        numeric_metrics[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
                mlflow.log_metrics(numeric_metrics)
        except Exception as exc:  # pragma: no cover - defensive, see module docstring
            print(f"[ExperimentLogger] MLflow dual-write skipped for run {run.run_id}: {exc}")

    def _append_csv_row(self, row: dict[str, Any]) -> None:
        row_df = pd.DataFrame([row])
        if self.csv_path.exists():
            existing = pd.read_csv(self.csv_path)
            combined = pd.concat([existing, row_df], ignore_index=True)
        else:
            combined = row_df
        combined.to_csv(self.csv_path, index=False)

    def _write_json(self, run: ExperimentRun) -> None:
        path = self.detail_dir / f"{run.run_id}.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "run_id": run.run_id,
                    "experiment": run.experiment,
                    "model_name": run.model_name,
                    "started_at": run.started_at,
                    "ended_at": run.ended_at,
                    "params": run.params,
                    "metrics": run.metrics,
                    "tags": run.tags,
                },
                f,
                indent=2,
                default=str,
            )
