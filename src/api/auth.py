"""
Phase 8 — Minimal auth (Section 19): "no RBAC/multi-role auth for the
MVP — a single API key (or simple email/password login) is enough to
demonstrate an authenticated endpoint without building a full
identity system."

Resolved shape: a per-user API key, not one shared/global key —
needed so `GET /checkins` (Section 19: "list own history") has a real
"own" to filter by. Registration takes only an email; the server
generates the key and returns it once. No password, no hashing, no
JWT/session machinery — the smallest credential that still gives each
user a distinct, checkable identity.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.db import get_db
from src.api.db_models import User, generate_api_key
from src.api.schemas import RegisterRequest, RegisterResponse

router = APIRouter()


@router.post("/auth/register", response_model=RegisterResponse, status_code=201)
def register(request: RegisterRequest, db: Session = Depends(get_db)) -> RegisterResponse:
    existing = db.execute(
        select(User).where(User.email == request.email)
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email already registered.")

    user = User(email=request.email, api_key=generate_api_key())
    db.add(user)
    db.commit()
    db.refresh(user)
    return RegisterResponse(email=user.email, api_key=user.api_key)


def get_current_user(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> User:
    """Dependency for every protected route. FastAPI turns a missing
    or invalid key into a clean 401 — mirrors Phase 7's "malformed
    input never touches real logic" pattern in schemas.py, just for
    auth instead of input validation."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

    user = db.execute(select(User).where(User.api_key == x_api_key)).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return user
