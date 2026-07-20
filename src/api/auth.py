"""
src/api/auth.py — API key authentication for LEX-DISCOVERY.

Usage:
    @router.post("/endpoint")
    async def my_endpoint(request: Request, _: None = Depends(require_api_key)):
        ...

The API key is expected in the X-API-Key HTTP header.
Keys are compared via SHA-256 hash to avoid timing attacks and plaintext logging.
"""

from __future__ import annotations

import hashlib

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from src.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _sha256_prefix(key: str) -> str:
    """Return first 12 hex chars of SHA-256(key) — safe to log."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


async def require_api_key(
    request: Request,
    api_key: str | None = Security(_API_KEY_HEADER),
) -> None:
    """
    FastAPI dependency: validates the X-API-Key header.
    Raises HTTP 401 if missing, HTTP 403 if invalid.
    """
    if not api_key:
        log.warning(
            "auth_missing_key",
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key not in settings.allowed_api_keys:
        log.warning(
            "auth_invalid_key",
            key_hash=_sha256_prefix(api_key),
            path=request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    log.debug(
        "auth_ok",
        key_hash=_sha256_prefix(api_key),
        path=request.url.path,
    )
