"""
Phase 8 — Database/Application Layer (Section 20), engine/session setup.

DB choice (resolved decision): SQLite through Phase 9 — Section 34
lists it as the "simpler alternative" to Postgres, adequate for
local-dev/portfolio scale. Migrated to Postgres at Phase 10
(Section 22's docker-compose `db: postgres` service): confirmed as
purely the config change this file was built for — `DATABASE_URL`
pointed at a real local Postgres, `init_db()` created all 4 tables
unchanged, and a live server driven against it round-tripped
register/check-in/history/per-user-isolation with zero changes to
this file or `db_models.py`. Local/non-Docker runs still default to
SQLite (below) when `DATABASE_URL` is unset; the Docker image (see
`../../docker-compose.yml`) always sets it to Postgres.

`DATABASE_URL` is read from the environment first, falling back to a
local SQLite file only if it's unset — this is what made the Postgres
migration a connection-string change rather than a schema rewrite.
The ORM models in `db_models.py` stick to portable SQLAlchemy column
types (Integer, String, Float, DateTime, JSON, ForeignKey) for the
same reason — nothing SQLite-specific.

No Alembic/migration framework yet — outside Section 20/34's stated
MVP scope; `init_db()` just creates tables that don't exist yet. A
real migration tool is a natural addition once the schema starts
changing post-MVP, not before.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Gitignored like mlflow.db (Phase 6) — each environment (your machine,
# this sandbox, later a container) grows its own local file rather
# than one being shipped inside a zip/commit.
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "wellpulse_app.db"
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")

# SQLite-only: a single file's connection is otherwise restricted to
# the thread that created it, which breaks under FastAPI's threadpool.
# Postgres (Phase 10) won't need this argument at all.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Shared declarative base for every ORM model in db_models.py."""


def get_db() -> Session:
    """FastAPI dependency: one Session per request, always closed
    afterward (commit/rollback is the caller's responsibility)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Creates the users/check_ins/predictions/prediction_explanations
    tables if they don't already exist. Called once at FastAPI
    startup (see main.py's lifespan) — safe to call repeatedly."""
    from src.api import db_models  # noqa: F401 — registers models on Base before create_all

    Base.metadata.create_all(bind=engine)
