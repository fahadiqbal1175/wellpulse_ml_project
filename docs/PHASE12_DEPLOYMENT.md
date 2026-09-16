# Phase 12 — Deployment (Section 24)

**Host:** Render (free tier) — chosen over Railway/Fly.io per Section 24's
"free/low-cost platform" framing; no Kubernetes, no over-engineering.

**Topology:**
- 1 Web Service (Docker, built from the repo's existing multi-stage
  Dockerfile — no code or Dockerfile changes were needed; `DATABASE_URL`
  is read from the environment per Phase 8/10's design)
- 1 Render Postgres (Free), same region as the web service, connected
  via Render's private network (Internal Database URL)

**Verified:** `GET /health` responds 200 with `model_loaded: true` from
the live `https://<service>.onrender.com` URL (not just locally); one
real `/api/v1/predict` call and one full register → check-in → history
round trip confirmed against the live instance.

**Known limitations (documented honestly, not glossed over):**
- Free web service: 512 MB RAM / 0.1 CPU, spins down after 15 min
  idle, ~30-60s cold start on the next request. Not production-grade;
  fine for a portfolio demo.
- Free Postgres: expires 30 days after creation, 14-day grace period
  to upgrade before deletion, no backups. Will need manual renewal or
  an upgrade to keep check-in history beyond that window.
- No staging environment / manual-approval deploy gate (Section 23's
  CI diagram) — out of scope for a single free-tier instance.

**Rollback:** unchanged from Section 28 — move the MLflow `Production`
alias back to the prior version and restart the Render service. No
deployment-target-specific rollback steps were needed.