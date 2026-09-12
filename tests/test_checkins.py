"""
Phase 8 verification (roadmap Phase 8: "a submitted check-in
round-trips through the DB"; Section 19: authenticated `/checkins`;
Section 20: check_ins/predictions/prediction_explanations
persistence).

DB isolation: each test gets its own fresh, temp-file SQLite database
— `src.api.db`'s module-level `engine`/`SessionLocal` are monkeypatched
onto it *before* the app's lifespan runs `init_db()` (TestClient's
`with` block triggers lifespan), so no test ever touches the real dev
`wellpulse_app.db`, and tests never see each other's users or
check-ins. (`init_db()`/`get_db()` both look up `engine`/`SessionLocal`
from the module at call time, so monkeypatching the module attribute
is enough — no dependency_overrides needed.)

MLflow isolation: like test_api.py, this module overrides
tests/conftest.py's autouse `_isolate_mlflow_tracking_store` with a
no-op — `/checkins` calls the real `ModelService`, which needs the
real registered Production model, not an empty throwaway registry.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import db as db_module
from src.api.main import app

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


@pytest.fixture(autouse=True)
def _isolate_mlflow_tracking_store():
    """Overrides tests/conftest.py's autouse fixture of the same name
    — this module intentionally talks to the real project mlflow.db /
    mlruns/ (read-only; submitting a check-in never writes to the
    registry)."""
    yield


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_wellpulse_app.db"
    test_engine = create_engine(
        f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}
    )
    test_session_local = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)

    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(db_module, "SessionLocal", test_session_local)

    with TestClient(app) as c:
        yield c


def _register(client: TestClient, email: str = "student@example.com") -> str:
    resp = client.post("/auth/register", json={"email": email})
    assert resp.status_code == 201
    return resp.json()["api_key"]


def test_register_returns_unique_api_key(client):
    key_a = _register(client, "a@example.com")
    key_b = _register(client, "b@example.com")
    assert key_a != key_b


def test_register_duplicate_email_returns_409(client):
    _register(client, "dupe@example.com")
    resp = client.post("/auth/register", json={"email": "dupe@example.com"})
    assert resp.status_code == 409


def test_checkins_requires_auth(client):
    resp = client.post("/checkins", json=VALID_PAYLOAD)
    assert resp.status_code == 401


def test_checkins_rejects_invalid_key(client):
    resp = client.post(
        "/checkins", json=VALID_PAYLOAD, headers={"X-API-Key": "not-a-real-key"}
    )
    assert resp.status_code == 401


def test_checkin_round_trips_through_the_db(client):
    """The Phase 8 milestone, verified literally: submit -> persisted
    -> read back via GET /checkins/{id} with the exact same content."""
    api_key = _register(client)
    headers = {"X-API-Key": api_key}

    submit_resp = client.post("/checkins", json=VALID_PAYLOAD, headers=headers)
    assert submit_resp.status_code == 201
    submitted = submit_resp.json()
    assert 1.0 <= submitted["predicted_score"] <= 10.0
    assert submitted["risk_tier"] in {"low_risk", "medium_risk", "high_risk"}
    assert len(submitted["top_factors"]) == 3

    fetch_resp = client.get(f"/checkins/{submitted['id']}", headers=headers)
    assert fetch_resp.status_code == 200
    assert fetch_resp.json() == submitted


def test_list_checkins_returns_own_history_newest_first(client):
    api_key = _register(client)
    headers = {"X-API-Key": api_key}

    first = client.post("/checkins", json=VALID_PAYLOAD, headers=headers).json()
    second_payload = {**VALID_PAYLOAD, "Avg_Daily_Usage_Hours": 8.0}
    second = client.post("/checkins", json=second_payload, headers=headers).json()

    list_resp = client.get("/checkins", headers=headers)
    assert list_resp.status_code == 200
    ids_in_order = [row["id"] for row in list_resp.json()]
    assert ids_in_order == [second["id"], first["id"]]


def test_checkins_are_isolated_per_user(client):
    key_a = _register(client, "alice@example.com")
    key_b = _register(client, "bob@example.com")

    client.post("/checkins", json=VALID_PAYLOAD, headers={"X-API-Key": key_a})

    bob_list = client.get("/checkins", headers={"X-API-Key": key_b})
    assert bob_list.status_code == 200
    assert bob_list.json() == []


def test_cannot_fetch_another_users_checkin(client):
    key_a = _register(client, "alice2@example.com")
    key_b = _register(client, "bob2@example.com")

    alice_checkin = client.post(
        "/checkins", json=VALID_PAYLOAD, headers={"X-API-Key": key_a}
    ).json()

    resp = client.get(f"/checkins/{alice_checkin['id']}", headers={"X-API-Key": key_b})
    assert resp.status_code == 404


def test_get_nonexistent_checkin_returns_404(client):
    api_key = _register(client)
    resp = client.get("/checkins/99999", headers={"X-API-Key": api_key})
    assert resp.status_code == 404


def test_checkins_malformed_input_returns_422_and_persists_nothing(client):
    api_key = _register(client)
    headers = {"X-API-Key": api_key}
    payload = {**VALID_PAYLOAD, "Age": 5}  # below ge=10

    resp = client.post("/checkins", json=payload, headers=headers)
    assert resp.status_code == 422

    list_resp = client.get("/checkins", headers=headers)
    assert list_resp.json() == []
