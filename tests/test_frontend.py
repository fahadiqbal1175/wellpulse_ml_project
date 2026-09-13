"""
Phase 9 verification (Section 21's frontend, mounted per Section 34's
plain-HTML/JS decision).

Keeps to the blueprint's testing scope (Section 29 lists a "post-deploy
smoke test hitting /health and one real prediction" for deployment,
nothing frontend-specific) — this is the same idea applied to the
static mount: confirm it actually serves the frontend, and confirm
mounting it at "/" didn't shadow any existing API route. It does not
exercise app.js itself; that was verified separately with a real
browser DOM (jsdom) driving the live app against a running server —
see the Phase 9 implementation guide.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root_serves_frontend_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "WellPulse" in resp.text


def test_static_assets_are_served():
    css = client.get("/style.css")
    js = client.get("/app.js")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]


def test_static_mount_does_not_shadow_api_routes():
    # Mounted last, at "/" — this only means anything if /health (an
    # existing exact-path route registered earlier) still resolves to
    # the health router and not a 404 from StaticFiles.
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "model_loaded" in resp.json()


def test_unknown_path_returns_404_not_index_fallback():
    # html=True enables an index.html fallback for "/", not a SPA
    # catch-all for arbitrary unmatched paths — there's no
    # client-side router here to hand unknown paths to.
    resp = client.get("/this-route-does-not-exist")
    assert resp.status_code == 404
