# ─────────────────────────────────────────────────────────────────────────────
# LEX-DISCOVERY Makefile
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: install install-full chainlit serve serve-chainlit test test-unit test-integration lint \
        ingest ingest-upload graph-diagram clean help

# ── Variables ──────────────────────────────────────────────────────────────
PYTHON      := python3
PIP         := pip install
UVICORN     := uvicorn src.api.main:app
PORT        := 8000
CHAINLIT_PORT := 8080
DB_PATH     := data/checkpoints.db

# ── Installation ───────────────────────────────────────────────────────────
install:
	@echo "📦 Installing core dependencies..."
	$(PIP) -e ".[dev]"
	@echo "✅ Core dependencies installed."

install-full:
	@echo "📦 Installing all dependencies (including LlamaIndex ingestion)..."
	$(PIP) -e ".[dev,ingestion]"
	@echo "✅ All dependencies installed."

# ── Serve ──────────────────────────────────────────────────────────────────
serve:
	@echo "🚀 Starting LEX-DISCOVERY API on port $(PORT)..."
	$(UVICORN) --host 0.0.0.0 --port $(PORT) --reload

serve-prod:
	@echo "🚀 Starting in production mode..."
	$(UVICORN) --host 0.0.0.0 --port $(PORT) --workers 4

chainlit:
	@echo "⚖️  Starting LEX-DISCOVERY Chainlit UI on port $(CHAINLIT_PORT)..."
	chainlit run chainlit_app.py --port $(CHAINLIT_PORT) --host 0.0.0.0

chainlit-dev:
	@echo "⚖️  Starting Chainlit in dev (hot-reload) mode..."
	chainlit run chainlit_app.py --port $(CHAINLIT_PORT) --host 0.0.0.0 -w

# ── Testing ────────────────────────────────────────────────────────────────
test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v --tb=short

test-integration:
	@echo "🧪 Running integration tests (requires GOOGLE_API_KEY)..."
	pytest tests/integration/ -v -m integration --tb=short

test:
	@echo "🧪 Running all tests..."
	pytest tests/ -v --tb=short --ignore=tests/e2e

# ── Linting ────────────────────────────────────────────────────────────────
lint:
	@echo "🔍 Running ruff..."
	ruff check src/ tests/
	@echo "🔍 Running mypy..."
	mypy src/ --ignore-missing-imports || true

format:
	@echo "✨ Formatting with ruff..."
	ruff format src/ tests/

# ── Data / Ingestion ──────────────────────────────────────────────────────
ingest:
	@echo "📄 Ingesting documents from data/raw/..."
	$(PYTHON) scripts/ingest.py --dir data/raw

ingest-case-law:
	@echo "📄 Ingesting case law from data/case_law/..."
	$(PYTHON) scripts/ingest.py --dir data/case_law --collection case_law_precedents

# ── Graph ─────────────────────────────────────────────────────────────────
graph-diagram:
	@echo "🗺  Generating graph diagram..."
	$(PYTHON) scripts/visualise_graph.py

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning up..."
	rm -rf data/checkpoints.db data/memory/ data/uploads/*.pdf
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Clean."

# ── Help ──────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "LEX-DISCOVERY — Multi-Source Legal Discovery Graph"
	@echo "────────────────────────────────────────────────────"
	@echo "  make install         Install core + dev dependencies"
	@echo "  make install-full    Install all deps (incl. LlamaIndex)"
	@echo "  make serve           Start dev server (hot-reload)"
	@echo "  make serve-prod      Start production server (4 workers)"
	@echo "  make test-unit       Run fast unit tests"
	@echo "  make test-integration Run integration tests (needs GOOGLE_API_KEY)"
	@echo "  make test            Run all tests"
	@echo "  make lint            Ruff + mypy"
	@echo "  make format          Auto-format with ruff"
	@echo "  make ingest          Ingest docs from data/raw/"
	@echo "  make ingest-case-law Ingest case law PDFs into Qdrant"
	@echo "  make graph-diagram   Generate pipeline visualisation"
	@echo "  make clean           Remove checkpoints, uploads, __pycache__"
	@echo ""
