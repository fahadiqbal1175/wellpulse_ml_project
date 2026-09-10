"""
Phase 7 — Inference service (Section 18).

Loads the Production model exactly once per process (via the MLflow
registry alias, per the blueprint's own wording), caches it, and
serves single-request predictions with SHAP explanation + a
residual-based confidence interval — all by REUSING Phase 5/6 code,
never reimplementing it:

  - Feature prep: `add_engineered_features` + the SAME
    `CategoricalFeatureEncoder` instance that trained the model
    (Section 18: "same encoders/scalers used in training... the most
    common real-world source of training/serving skew").
  - SHAP: `compute_shap_values` / `top_n_factors` /
    `generate_explanation_sentence` / `CONFIDENCE_Z`, imported
    straight from `src/evaluation/explainability.py`.
  - Confidence interval: `compute_residual_std_on_val()`, called once
    at startup (it rebuilds the val split/predictions internally, so
    it's too expensive to call per-request — exactly why it's cached
    here rather than recomputed on every `/predict` call).
  - Risk tier: `compute_risk_tier()` from `src/features/engineering.py`.

One deliberate deviation from a literal reading of Section 18's
"encoders/scalers... loaded from the registry alongside the model":
`mlflow.sklearn.log_model()` (see `src/experiments/mlflow_utils.py`)
logs only the raw sklearn estimator, not a pyfunc wrapper bundling the
encoder — so the registry alone has no encoder to load. The encoder
instead comes from `models/final_model.joblib`, which
`register_final_model.py` guarantees is the same `(model, encoder)`
pair that got logged and registered in the same run — so this is
still "the one true encoder used in training," just sourced from the
joblib bundle instead of the registry artifact itself. Worth fixing
properly if this project ever moves to a pyfunc wrapper.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import shap
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.api.recommendations import get_recommendations
from src.evaluation.explainability import (
    CONFIDENCE_Z,
    compute_residual_std_on_val,
    generate_explanation_sentence,
    top_n_factors,
)
from src.evaluation.final_evaluation import FINAL_MODEL_PATH
from src.experiments.mlflow_utils import REGISTERED_MODEL_NAME, init_mlflow
from src.features.engineering import add_engineered_features, compute_risk_tier

logger = logging.getLogger(__name__)

PRODUCTION_ALIAS = "Production"


class ModelUnavailableError(RuntimeError):
    """Raised when a prediction is requested but no Production model
    is loaded — the API layer turns this into a 503 (Section 18:
    'missing/corrupt model artifact -> 503 with a clear message')."""


class ModelService:
    """One instance lives on `app.state` for the process lifetime.
    `load()` is called once at FastAPI startup (see `main.py`'s
    lifespan) and never again per request."""

    def __init__(self) -> None:
        self.model = None
        self.encoder = None
        self.feature_columns: list[str] | None = None
        self.model_name: str | None = None
        self.model_version: str | None = None
        self.explainer = None
        self.residual_std: float | None = None
        self.load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.encoder is not None

    def load(self, final_model_path: Path = FINAL_MODEL_PATH) -> None:
        try:
            init_mlflow()
            client = MlflowClient()
            try:
                mv = client.get_model_version_by_alias(
                    REGISTERED_MODEL_NAME, PRODUCTION_ALIAS
                )
            except MlflowException as exc:
                raise ModelUnavailableError(
                    f"No '{PRODUCTION_ALIAS}' alias registered for "
                    f"'{REGISTERED_MODEL_NAME}' — run "
                    f"`python -m src.experiments.register_final_model` first."
                ) from exc

            model = mlflow.sklearn.load_model(
                f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
            )

            if not final_model_path.exists():
                raise ModelUnavailableError(
                    f"{final_model_path} not found — run "
                    f"`python -m src.evaluation.final_evaluation` first."
                )
            bundle = joblib.load(final_model_path)

            # Assign only after every step above succeeds, so a
            # partial failure can never leave is_ready True with a
            # half-initialized service.
            self.model = model
            self.model_version = f"{REGISTERED_MODEL_NAME}:{mv.version}"
            self.encoder = bundle["encoder"]
            self.feature_columns = bundle["feature_columns"]
            self.model_name = bundle["model_name"]
            self.explainer = shap.TreeExplainer(self.model)
            self.residual_std = compute_residual_std_on_val()
            self.load_error = None
            logger.info(
                "Loaded Production model %s (version %s)",
                self.model_name,
                mv.version,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any
            # failure here must degrade to "service unavailable", never
            # crash app startup (Section 18's 503 path, not a 500 or a
            # dead process).
            self.model = None
            self.load_error = str(exc)
            logger.error("Model load failed: %s", exc)

    def _prepare_features(self, raw: dict) -> pd.DataFrame:
        """Mirrors `build_model_ready_xy`'s X-construction exactly
        (`src/models/baseline.py`), minus the target-column split — a
        live request has no `Mental_Health_Score`."""
        df = pd.DataFrame([raw])
        engineered = add_engineered_features(df)
        encoded = self.encoder.transform(engineered)
        return encoded[self.feature_columns]

    def predict_one(self, raw: dict) -> dict:
        if not self.is_ready:
            raise ModelUnavailableError(
                self.load_error or "Model not loaded."
            )

        X = self._prepare_features(raw)
        prediction = float(self.model.predict(X)[0])

        shap_values = self.explainer(X)
        factors = top_n_factors(shap_values.values[0], list(X.columns))
        sentence = generate_explanation_sentence(factors)
        risk_tier = str(
            compute_risk_tier(
                pd.DataFrame({"Mental_Health_Score": [prediction]})
            ).iloc[0]
        )
        recommendations = get_recommendations(factors)

        return {
            "predicted_score": round(prediction, 3),
            "risk_tier": risk_tier,
            "confidence_interval_68pct": [
                round(prediction - CONFIDENCE_Z * self.residual_std, 3),
                round(prediction + CONFIDENCE_Z * self.residual_std, 3),
            ],
            "top_factors": factors,
            "explanation_sentence": sentence,
            "recommendations": recommendations,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }
