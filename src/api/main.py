"""
src/api/main.py — FastAPI application factory with full middleware stack and lifespan.

Startup sequence:
  1. setup_logging()        — structlog JSON/console
  2. setup_langsmith()      — opt-in LangSmith tracing
  3. setup_llamaindex()     — HuggingFace embeddings init
  4. get_app_async()        — compile LangGraph with AsyncSqliteSaver
  5. Register middleware:
       - RequestIDMiddleware   (X-Request-ID)
       - SecurityHeadersMiddleware
       - CORSMiddleware
       - SlowAPIMiddleware (rate limiting)
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import router
from src.config import cfg, settings
from src.core.logging import get_logger, setup_logging
from src.core.tracing import setup_langsmith

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────
_rate_limit = cfg.get("api", {}).get("rate_limit", "30/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[_rate_limit])


# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID to every request and bind it to structlog context."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Inject X-Process-Time (milliseconds) into every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{elapsed_ms:.1f}ms"
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        return response


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown hooks."""
    # 1. Logging
    setup_logging(app_env=settings.app_env)
    log.info("lex_discovery_starting", env=settings.app_env)

    # 2. LangSmith tracing (opt-in)
    setup_langsmith(
        api_key=settings.langsmith_api_key,
        project=settings.langsmith_project,
    )

    # 3. LlamaIndex embeddings
    try:
        from src.core.llamaindex_setup import setup_llamaindex

        setup_llamaindex(google_api_key=settings.google_api_key)
    except Exception as exc:
        log.warning("llamaindex_setup_warning", error=str(exc))

    # 4. Compile LangGraph with async checkpointer
    try:
        from src.graph.graph import get_app_async

        app.state.graph_app = await get_app_async()
        log.info("langgraph_compiled")
    except Exception as exc:
        log.error("langgraph_compile_error", error=str(exc))
        app.state.graph_app = None

    # 5. Ensure data directories exist
    import pathlib

    for dir_name in ("data/uploads", "data/memory", "data"):
        pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

    log.info("lex_discovery_started")
    yield

    # Shutdown
    log.info("lex_discovery_shutdown")


# ─────────────────────────────────────────────────────────────────────────────
# App Factory
# ─────────────────────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="LEX-DISCOVERY — Multi-Source Legal Discovery Graph",
        description=(
            "A 5-stage LangGraph pipeline for legal discovery: "
            "PDF lease extraction → Qdrant precedent search → "
            "Cross-context gap analysis → Human-in-the-Loop review → Verdict generation."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # Security & observability headers
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # CORS
    cors_origins = cfg.get("api", {}).get("cors_origins", ["http://localhost:3000"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root health/version endpoints (infrastructure probes, no API prefix)
    @app.get("/health", tags=["System"])
    async def root_health() -> dict[str, str]:
        return {"status": "ok", "app_env": settings.app_env}

    @app.get("/version", tags=["System"])
    async def root_version() -> dict[str, str]:
        return {"version": "1.0.0", "pipeline": "Multi-Source Legal Discovery Graph"}

    # Routes — all under /api/v1 prefix
    app.include_router(router, prefix="/api/v1")

    # Global error handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error(
            "unhandled_exception",
            error=str(exc),
            path=str(request.url),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."},
        )

    return app


app = create_app()
