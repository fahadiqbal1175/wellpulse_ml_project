"""
Phase 7 verification (Section 37's "a real HTTP request returns a
score + explanation", Milestone ML-6).

Note on MLflow isolation: `tests/conftest.py`'s autouse
`_isolate_mlflow_tracking_store` fixture points every test at a
throwaway, empty sqlite db — correct for training-code tests, but
wrong here: the whole point of `test_predict_valid_input_*` is to
load the REAL registered Production model and get a real prediction
from it (matching this project's "actually run it against the real
repo, don't simulate" practice from every earlier phase). This file
overrides that fixture with a no-op of the same name so it applies to
every test in this module — the 503 test below achieves its "no
model" scenario via explicit monkeypatching instead, not by relying
on isolation to make the registry empty.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import inference as inference_module
from src.api.main import app


@pytest.fixture(autouse=True)
def _isolate_mlflow_tracking_store():
    """Overrides tests/conftest.py's autouse fixture of the same name
    — this module intentionally talks to the real project mlflow.db /
    mlruns/ (read-only; predicting never writes to the registry)."""
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "Age": 20,
    "Gender": "Female",
    "Academic_Level": "Undergraduate",
    "Country": "Pakistan",
    "Avg_Daily_Usage_Hours": 5.2,
    "Most_Used_Platform": "Instagram",
    "Sleep_Hours_Per_Night": 6.0,
    "Relationship_Status": "Single",
    "Conflicts_Over_Social_Media": 3,
}


def test_health_reports_real_production_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "wellpulse_final_model:1"


def test_predict_valid_input_returns_score_and_explanation(client):
    resp = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()

    assert 1.0 <= body["predicted_score"] <= 10.0
    assert body["risk_tier"] in {"low_risk", "medium_risk", "high_risk"}
    assert len(body["confidence_interval_68pct"]) == 2
    assert body["confidence_interval_68pct"][0] < body["confidence_interval_68pct"][1]
    assert len(body["top_factors"]) == 3
    for factor in body["top_factors"]:
        assert factor["direction"] in {"raising", "lowering"}
    assert body["explanation_sentence"].startswith("This prediction was most shaped by")
    assert isinstance(body["recommendations"], list)
    assert body["model_name"] in {"Random Forest (tuned)", "Decision Tree (untuned)"}
    assert body["model_version"] == "wellpulse_final_model:1"
    assert "not a" in body["disclaimer"].lower()


def test_predict_high_usage_low_sleep_high_conflict_skews_toward_lower_score(client):
    """Not a hard guarantee of monotonicity for every case, but a real
    end-to-end sanity check against the ACTUAL model: an extreme
    high-usage/low-sleep/high-conflict profile should not land in the
    same territory as a comfortable, well-rested, low-conflict one."""
    strained = {**VALID_PAYLOAD, "Avg_Daily_Usage_Hours": 9.5,
                "Sleep_Hours_Per_Night": 4.0, "Conflicts_Over_Social_Media": 5}
    comfortable = {**VALID_PAYLOAD, "Avg_Daily_Usage_Hours": 1.5,
                   "Sleep_Hours_Per_Night": 9.0, "Conflicts_Over_Social_Media": 0}

    strained_score = client.post("/api/v1/predict", json=strained).json()["predicted_score"]
    comfortable_score = client.post("/api/v1/predict", json=comfortable).json()["predicted_score"]

    assert strained_score < comfortable_score


@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("Age", 5),  # below ge=10
        ("Gender", "Nonbinary"),  # not in KNOWN_GENDERS
        ("Avg_Daily_Usage_Hours", -1),  # below ge=0
        ("Most_Used_Platform", "MySpace"),  # not in KNOWN_PLATFORMS
        ("Sleep_Hours_Per_Night", 0),  # gt=0, not ge=0 (div-by-zero guard)
        ("Conflicts_Over_Social_Media", 25),  # above le=20
    ],
)
def test_predict_malformed_input_returns_422_before_touching_model(
    client, monkeypatch, bad_field, bad_value
):
    called = {"predict": False}

    def _fail_if_called(self, raw):
        called["predict"] = True
        raise AssertionError("Model must never be touched for malformed input")

    monkeypatch.setattr(inference_module.ModelService, "predict_one", _fail_if_called)

    payload = {**VALID_PAYLOAD, bad_field: bad_value}
    resp = client.post("/api/v1/predict", json=payload)

    assert resp.status_code == 422
    assert called["predict"] is False


def test_predict_missing_model_returns_503(client, monkeypatch):
    service = app.state.model_service
    original_model, original_error = service.model, service.load_error
    monkeypatch.setattr(service, "model", None)
    monkeypatch.setattr(service, "load_error", "simulated missing model artifact")

    resp = client.post("/api/v1/predict", json=VALID_PAYLOAD)

    assert resp.status_code == 503
    assert "simulated missing model artifact" in resp.json()["detail"]

    # restore, since `service` is the shared app-lifetime instance
    monkeypatch.setattr(service, "model", original_model)
    monkeypatch.setattr(service, "load_error", original_error)


def test_health_reports_unavailable_when_model_missing(client, monkeypatch):
    service = app.state.model_service
    monkeypatch.setattr(service, "model", None)
    monkeypatch.setattr(service, "load_error", "simulated missing model artifact")

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unavailable"
    assert body["model_loaded"] is False
    assert body["detail"] == "simulated missing model artifact"
