# ─────────────────────────────────────────────────────────────────────────────
# LEX-DISCOVERY — Multi-Stage Dockerfile
#
# Stages:
#   1. base        — shared Python slim base with system deps
#   2. builder     — installs Python packages into a virtual-env (build cache)
#   3. api         — production FastAPI/Uvicorn runtime  (default target)
#   4. chainlit    — production Chainlit UI runtime
#   5. dev         — development image with hot-reload + dev extras
#
# Build examples:
#   docker build --target api     -t lex-discovery:api .
#   docker build --target chainlit -t lex-discovery:chainlit .
#   docker build --target dev      -t lex-discovery:dev .
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.11
ARG APP_USER=appuser
ARG APP_UID=1001

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — base: slim Python + OS packages
# ══════════════════════════════════════════════════════════════════════════════
FROM python:${PYTHON_VERSION}-slim AS base

ARG APP_USER
ARG APP_UID

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# OS-level dependencies (curl for health-check, libmagic for file type detection)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libmagic1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user created once; reused across runtime stages
RUN groupadd --gid ${APP_UID} ${APP_USER} \
    && useradd  --uid ${APP_UID} --gid ${APP_UID} \
                --shell /bin/bash --create-home ${APP_USER}

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — builder: install all Python deps into an isolated venv
# ══════════════════════════════════════════════════════════════════════════════
FROM base AS builder

WORKDIR /build

# Create venv inside the build stage so it can be COPY'd to runtime images
RUN python -m venv ${VIRTUAL_ENV}

# Copy only the files needed to resolve dependencies first (maximises cache)
COPY pyproject.toml ./

# Install core project deps (no editable install — we copy src directly)
RUN pip install --upgrade pip setuptools wheel \
    && pip install ".[dev]"

# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — api: production FastAPI/Uvicorn image  (default build target)
# ══════════════════════════════════════════════════════════════════════════════
FROM base AS api

ARG APP_USER
ARG APP_UID

LABEL org.opencontainers.image.title="LEX-DISCOVERY API" \
      org.opencontainers.image.description="Multi-Source Legal Discovery Graph — FastAPI Runtime" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Copy the pre-built venv from builder (no pip invocation at runtime)
COPY --from=builder --chown=${APP_USER}:${APP_USER} ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy application source
COPY --chown=${APP_USER}:${APP_USER} src/         ./src/
COPY --chown=${APP_USER}:${APP_USER} configs/     ./configs/
COPY --chown=${APP_USER}:${APP_USER} prompts/     ./prompts/
COPY --chown=${APP_USER}:${APP_USER} scripts/     ./scripts/
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml ./

# Persistent data directories (mounted as volumes in production)
RUN mkdir -p data/uploads data/memory data/checkpoints \
    && chown -R ${APP_USER}:${APP_USER} data/

USER ${APP_USER}

EXPOSE 8000

# Liveness probe endpoint — FastAPI /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production: 4 Uvicorn workers (override via CMD / docker-compose)
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "${PORT:-8000}", \
     "--workers", "1", \
     "--access-log", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]

# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — chainlit: production Chainlit UI image
# ══════════════════════════════════════════════════════════════════════════════
FROM base AS chainlit

ARG APP_USER
ARG APP_UID

LABEL org.opencontainers.image.title="LEX-DISCOVERY Chainlit UI" \
      org.opencontainers.image.description="Multi-Source Legal Discovery Graph — Chainlit Runtime" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY --from=builder --chown=${APP_USER}:${APP_USER} ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY --chown=${APP_USER}:${APP_USER} src/            ./src/
COPY --chown=${APP_USER}:${APP_USER} configs/        ./configs/
COPY --chown=${APP_USER}:${APP_USER} prompts/        ./prompts/
COPY --chown=${APP_USER}:${APP_USER} chainlit_app.py ./
COPY --chown=${APP_USER}:${APP_USER} chainlit.md     ./
COPY --chown=${APP_USER}:${APP_USER} .chainlit/      ./.chainlit/
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml  ./

RUN mkdir -p data/uploads data/memory data/checkpoints \
    && chown -R ${APP_USER}:${APP_USER} data/

USER ${APP_USER}

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

CMD ["chainlit", "run", "chainlit_app.py", \
     "--host", "0.0.0.0", \
     "--port", "8080"]

# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — dev: hot-reload development image (not for production)
# ══════════════════════════════════════════════════════════════════════════════
FROM builder AS dev

LABEL org.opencontainers.image.title="LEX-DISCOVERY Dev" \
      org.opencontainers.image.description="Development image with hot-reload"

WORKDIR /app

# Mount the project root as a volume for hot-reload (see docker-compose.dev.yml)
COPY . .

# Install the project in editable mode so code changes are reflected instantly
RUN pip install -e ".[dev]"

EXPOSE 8000 8080

# Default: API with hot-reload (override to run chainlit)
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload"]
