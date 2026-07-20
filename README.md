<div align="center">

# ⚖️ LEX-DISCOVERY

**Multi-Source Legal Discovery Graph**

*A production-grade AI pipeline for automated legal discovery, compliance gap analysis, and verdict generation — built for Kenyan jurisdiction.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-DC244C?style=flat-square)](https://qdrant.tech)
[![Chainlit](https://img.shields.io/badge/Chainlit-UI-FF6B35?style=flat-square)](https://chainlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Vector Store](#vector-store)
- [Ingestion](#ingestion)
- [Chainlit UI](#chainlit-ui)
- [Testing](#testing)
- [Development](#development)

---

## Overview

LEX-DISCOVERY automates the labour-intensive phases of legal discovery by orchestrating a multi-agent LangGraph pipeline that:

1. **Extracts** structured data from client documents (lease agreements, contracts, correspondence)
2. **Retrieves** relevant case law precedents via hybrid semantic + BM25 search against Qdrant Cloud
3. **Analyses** compliance gaps between the client's situation and applicable statutes / precedents
4. **Surfaces** findings to lead counsel for human-in-the-loop (HITL) review
5. **Generates** a formal legal verdict memo on approval

The system targets **Kenyan tenancy law** out-of-the-box but is jurisdiction-agnostic at the prompt layer.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT INTERFACES                            │
│   Chainlit UI (port 8080)        FastAPI REST (port 8000)           │
└────────────────────┬────────────────────────┬───────────────────────┘
                     │                        │
                     ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH MAIN GRAPH                             │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐        │
│  │   Lead       │    │ Client Files │    │   Case Law      │        │
│  │  Attorney   ├───►│  Subgraph    ├───►│   Subgraph      │        │
│  │ Supervisor  │    │              │    │                 │        │
│  └─────────────┘    └──────────────┘    └────────┬────────┘        │
│                                                  │                  │
│  ┌───────────────────┐    ┌──────────┐           │                  │
│  │  Cross-Context    │◄───┤          │◄──────────┘                  │
│  │  Gap Analysis     │    │  State   │                              │
│  └────────┬──────────┘    └──────────┘                              │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐    ┌──────────────────┐                       │
│  │  Human Review   │    │  Generate Verdict │                       │
│  │  (HITL/pause)  ├───►│  (on approval)   │                       │
│  └─────────────────┘    └──────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                    │
│                                                                     │
│  Qdrant Cloud                    SQLite Checkpointer                │
│  ┌──────────────────────────┐    ┌────────────────────────┐        │
│  │ Dense:  bge-small-en     │    │  data/checkpoints.db   │        │
│  │         (dim=384)        │    │  (LangGraph state)     │        │
│  │ Sparse: Qdrant/bm25      │    └────────────────────────┘        │
│  │ Fusion: RRF              │                                       │
│  └──────────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Walkthrough

The pipeline is a directed acyclic graph (DAG) with conditional routing at every stage. Each node has a graceful fallback path to `rejection → END`.

```
lead_attorney_ingestion
        │
        ├── (no matter) ──► rejection ──► END
        │
        ▼
client_files_runner  [Client Files subgraph]
        │
        ├── (no data) ──► rejection ──► END
        │
        ▼
case_law_runner  [Case Law subgraph → Qdrant hybrid search]
        │
        ├── (error) ──► rejection ──► END
        │
        ▼
cross_context_mapping  [LLM gap analysis]
        │
        ▼ (always)
human_review  ← ── ── ── ── PAUSE (LangGraph interrupt)
        │                         ▲
        │                  POST /discovery/{id}/approve
        │
        ├── (rejected) ──► rejection ──► END
        │
        ▼
generate_verdict  [LLM verdict memo]
        │
        ▼
       END
```

### Nodes

| Node | Description |
|------|-------------|
| `lead_attorney_ingestion` | Validates client matter, formulates legal hypothesis via LLM |
| `client_files_runner` | Invokes the Client Files subgraph; extracts timeline & clauses from PDFs |
| `case_law_runner` | Invokes the Case Law subgraph; retrieves precedents from Qdrant |
| `cross_context_mapping` | LLM-powered compliance gap analysis (timeline vs precedents) |
| `human_review` | **HITL interrupt** — suspends graph pending lead counsel decision |
| `generate_verdict` | Generates formal legal verdict memo (Executive Summary → Actions) |
| `rejection_node` | Terminates pipeline gracefully on invalid input or counsel rejection |

### Subgraphs

| Subgraph | Purpose |
|----------|---------|
| `client_files` | PDF loading → clause extraction → timeline construction |
| `case_law` | Hypothesis → Qdrant hybrid search → ranked precedents |

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Orchestration** | LangGraph ≥ 0.2 | Stateful graph, HITL interrupts, SQLite checkpointing |
| **LLM — Primary** | Saul-7B-Instruct-v1 (HuggingFace) | Law-specialised legal reasoning |
| **LLM — Fallback** | Gemini 2.0 Flash | Fallback when HF inference is unavailable |
| **Vector Store** | Qdrant Cloud | Hybrid dense + sparse (BM25) retrieval |
| **Embeddings** | BAAI/bge-small-en-v1.5 (fastembed) | Dense vectors, dim=384 |
| **Sparse Encoding** | Qdrant/bm25 (fastembed) | BM25 sparse vectors, no external tokeniser |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Native Qdrant hybrid scoring |
| **RAG Framework** | LlamaIndex ≥ 0.12 | Vector store abstraction, ingestion pipeline |
| **API** | FastAPI + Uvicorn | Async REST API, rate-limited, API-key auth |
| **UI** | Chainlit | Chat interface for interactive discovery |
| **Observability** | LangSmith + structlog | Tracing, structured JSON logging |
| **Config** | Pydantic Settings + YAML | Layered config (YAML + env vars) |
| **State Persistence** | aiosqlite / SqliteSaver | Per-thread checkpoint store |

---

## Project Structure

```
LEX-DISCOVERY/
├── src/
│   ├── api/
│   │   ├── auth.py               # API key authentication
│   │   ├── main.py               # FastAPI app + lifespan
│   │   ├── routes.py             # All REST endpoints
│   │   └── schemas.py            # Pydantic request/response models
│   │
│   ├── core/
│   │   ├── exceptions.py         # Custom exception types
│   │   ├── llamaindex_setup.py   # LlamaIndex global settings (embed model, LLM)
│   │   ├── llm_factory.py        # LLM provider factory (Saul → Gemini fallback)
│   │   ├── logging.py            # structlog JSON logger
│   │   ├── prompt_manager.py     # Prompt loading & hot-reload
│   │   └── tracing.py            # LangSmith callback setup
│   │
│   ├── graph/
│   │   ├── graph.py              # Main graph builder + get_app() / get_app_async()
│   │   ├── nodes.py              # All 7 node implementations
│   │   ├── edges.py              # Conditional routing functions
│   │   ├── state.py              # DiscoveryState TypedDict + sub-states
│   │   └── subgraphs/
│   │       ├── client_files.py   # Client document extraction subgraph
│   │       └── case_law.py       # Case law retrieval subgraph
│   │
│   ├── ingestion/
│   │   └── llamaindex_pipeline.py  # PDF → chunks → embed → Qdrant upsert
│   │
│   ├── vectordb/
│   │   ├── llamaindex_qdrant.py  # QdrantVectorStoreWrapper (primary, LlamaIndex-native)
│   │   └── qdrant_store.py       # Low-level QdrantClient wrapper (collection mgmt, BM25)
│   │
│   ├── tools/
│   │   └── knowledge_base.py     # Mock search fallback + knowledge base tools
│   │
│   ├── memory/                   # (reserved) Session memory layer
│   └── config.py                 # Settings (Pydantic) + cfg (YAML loader)
│
├── configs/
│   └── config.yaml               # All tunable parameters
│
├── prompts/                      # External prompt templates (YAML/text)
├── scripts/
│   ├── ingest.py                 # CLI: bulk ingest PDFs into Qdrant
│   └── visualise_graph.py        # CLI: render graph diagram
│
├── data/
│   ├── uploads/                  # Uploaded client documents
│   └── checkpoints.db            # LangGraph SQLite checkpoint store
│
├── tests/                        # pytest test suite
├── chainlit_app.py               # Chainlit UI entry point
├── pyproject.toml                # Project metadata + dependencies
└── .env.example                  # Environment variable template
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- A [Qdrant Cloud](https://cloud.qdrant.io) cluster (free tier works)
- A HuggingFace API key (for Saul-7B) **or** a Google API key (for Gemini fallback)

### 1 — Clone & install

```bash
git clone <repo-url>
cd LEX-DISCOVERY

# Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install core dependencies
pip install -e ".[dev]"

# Install ingestion extras (optional — only needed for PDF ingestion)
pip install -e ".[ingestion]"
```

### 2 — Configure environment

```bash
cp .env.example .env
# Edit .env with your actual API keys (see Environment Variables section)
```

### 3 — Start the FastAPI server

```bash
uvicorn src.api.main:app --reload --port 8000
```

API docs auto-generated at: **http://localhost:8000/docs**

### 4 — Start the Chainlit UI

```bash
chainlit run chainlit_app.py --port 8080
```

Chat interface at: **http://localhost:8080**

### 5 — Ingest case law (optional)

```bash
# Ingest a single PDF
python scripts/ingest.py --file data/uploads/my_case.pdf

# Ingest an entire directory
python scripts/ingest.py --dir data/case_law/
```

---

## Configuration

All tuneable parameters live in [`configs/config.yaml`](configs/config.yaml). Environment variables take precedence over YAML values.

```yaml
llm:
  default_model: "Equall/Saul-7B-Instruct-v1"   # Primary LLM
  provider: "huggingface"                         # huggingface | gemini
  temperature: 0.1
  fallback_model: "gemini-2.0-flash"              # Automatic fallback

qdrant:
  collection_name: "case_law_precedents"
  vector_size: 384                                # Must match bge-small-en-v1.5
  sparse_model: "Qdrant/bm25"
  dense_model: "BAAI/bge-small-en-v1.5"

retrieval:
  top_k: 5
  rerank: true                                    # Enable RRF re-ranking

ingestion:
  chunk_size: 512
  chunk_overlap: 64

graph:
  max_iterations: 10
  human_in_the_loop: true                         # HITL is mandatory

api:
  rate_limit: "30/minute"
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_API_KEY` | Yes* | HuggingFace API key (Saul-7B primary LLM) |
| `HF_MODEL_ID` | No | Override HF model ID (default: `Equall/Saul-7B-Instruct-v1`) |
| `GOOGLE_API_KEY` | Yes* | Google Gemini API key (fallback LLM) |
| `QDRANT_URL` | Yes | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud API key |
| `VECTOR_BACKEND` | No | `qdrant` (default) |
| `ALLOWED_API_KEYS` | Yes | JSON array of valid API keys, e.g. `["key1","key2"]` |
| `APP_ENV` | No | `development` (default) \| `production` |
| `LANGSMITH_API_KEY` | No | LangSmith observability API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name (default: `lex-discovery`) |
| `LANGCHAIN_TRACING_V2` | No | `true` to enable LangSmith tracing |
| `MEMORY_ENCRYPTION_KEY` | No | 32-byte base64 key for session memory encryption |

> *At least one of `HF_API_KEY` or `GOOGLE_API_KEY` must be set. If HuggingFace is unavailable, the system automatically falls back to Gemini.

---

## API Reference

All endpoints (except `/health` and `/version`) require the `X-API-Key` header.

### System

```
GET  /health    → HealthResponse      (no auth)
GET  /version   → VersionResponse     (no auth)
```

### Discovery Pipeline

#### Start a discovery session
```http
POST /discovery/start
X-API-Key: your-api-key
Content-Type: application/json

{
  "client_matter": "My landlord issued an eviction notice with only 18 days instead of the 30 required by the lease.",
  "file_path": "data/uploads/lease.pdf",   // optional
  "thread_id": "custom-id-123"             // optional, auto-generated if omitted
}
```

Response (`202 Accepted`):
```json
{
  "thread_id": "abc-123",
  "status": "AWAITING_REVIEW",
  "hypothesis": "Legal hypothesis: ...",
  "message": "Pipeline is paused at human review. Call POST /discovery/{thread_id}/approve ..."
}
```

#### Poll session status
```http
GET /discovery/{thread_id}
X-API-Key: your-api-key
```

Returns full state including timeline, case law results, compliance gaps, and final verdict if complete.

#### Submit human review decision (HITL resume)
```http
POST /discovery/{thread_id}/approve
X-API-Key: your-api-key
Content-Type: application/json

{
  "verdict_approved": true,
  "counsel_notes": "Findings are accurate. Proceed with filing."
}
```

Response: Final verdict memo on approval, or rejection message.

### Ingestion

#### Ingest from server path
```http
POST /ingest
X-API-Key: your-api-key
Content-Type: application/json

{
  "file_path": "data/uploads/case_law.pdf",
  "collection_name": "case_law_precedents"
}
```

#### Upload and ingest a PDF
```http
POST /ingest/upload
X-API-Key: your-api-key
Content-Type: multipart/form-data

file=@/path/to/document.pdf
collection_name=case_law_precedents
```

### Status Values

| Status | Description |
|--------|-------------|
| `INITIALISED` | Session created, pipeline starting |
| `RUNNING` | Pipeline actively processing |
| `AWAITING_REVIEW` | Paused at HITL node — awaiting counsel decision |
| `COMPLETE` | Verdict generated and approved |
| `REJECTED` | Terminated — invalid input or counsel rejection |

---

## Vector Store

The vector store layer has two implementations:

### `QdrantVectorStoreWrapper` (primary)

Located in [`src/vectordb/llamaindex_qdrant.py`](src/vectordb/llamaindex_qdrant.py). Uses LlamaIndex's native `QdrantVectorStore` abstraction — no manual BM25 tokenisation required.

```python
from src.vectordb import QdrantVectorStoreWrapper, get_wrapper

# Via factory (uses project config)
wrapper = get_wrapper()

# Hybrid search — dense + sparse via VectorStoreQuery(mode=HYBRID)
results = wrapper.hybrid_search(
    query_vector=[0.1, 0.2, ...],   # pre-computed dense embedding
    query_text="wrongful eviction notice period",
    top_k=5,
)
# → [{"id": "...", "score": 0.92, "payload": {...}}, ...]

# Upsert — wraps points as LlamaIndex TextNodes
wrapper.upsert([
    {
        "id": "doc-001",
        "vector": [0.1, 0.2, ...],
        "payload": {"text": "...", "title": "...", "citation": "..."},
    }
])
```

**How hybrid search works:**

```
query_text  ──► fastembed BM25 ──► sparse vector
query_vector ──────────────────► dense vector
                                       │
                         Qdrant RRF fusion
                                       │
                              ranked results
```

### `QdrantVectorStore` (low-level)

Located in [`src/vectordb/qdrant_store.py`](src/vectordb/qdrant_store.py). Direct `qdrant_client` wrapper used for collection management and the ingestion pipeline. Exposes `ensure_collection()`, `hybrid_search()`, and `upsert()` directly against the Qdrant API.

### Collection Schema

```
Collection: case_law_precedents
  Dense vector:  "" (unnamed)  — dim=384, Cosine distance
  Sparse vector: "bm25"        — SparseVectorParams
```

---

## Ingestion

The ingestion pipeline ([`src/ingestion/llamaindex_pipeline.py`](src/ingestion/llamaindex_pipeline.py)) processes PDF and text documents into Qdrant:

```
PDF / TXT file
      │
      ▼
SimpleDirectoryReader   (LlamaIndex)
      │
      ▼
SentenceSplitter        chunk_size=512, overlap=64
      │
      ▼
bge-small-en-v1.5       dense embedding (dim=384)
      │
      ▼
QdrantVectorStore       upsert with payload {text, doc_id, source, metadata}
```

**Security:** Path traversal protection — all ingestion paths are validated against `data/` as the allowed root. Set `skip_path_validation=True` only in tests.

```bash
# CLI ingestion
python scripts/ingest.py --file path/to/document.pdf --collection case_law_precedents
```

---

## Chainlit UI

The Chainlit UI (`chainlit_app.py`) provides a conversational interface that maps directly to the REST pipeline:

- Type a client matter → starts a discovery session
- Streams each node's output as it completes
- Presents compliance gaps for review
- Provides Approve / Reject action buttons (HITL)
- Displays the final verdict memo inline

```bash
chainlit run chainlit_app.py --port 8080
```

---

## Testing

```bash
# Run all tests
pytest

# Unit tests only (no external services)
pytest -m "not integration and not e2e"

# Integration tests (require API keys)
pytest -m integration

# With coverage
pytest --cov=src --cov-report=term-missing
```

Test markers:
- `integration` — requires `GOOGLE_API_KEY` and live Qdrant
- `e2e` — requires Playwright and a running server

---

## Development

### Code style

```bash
# Lint + format check
ruff check src/ tests/

# Auto-fix
ruff check --fix src/ tests/

# Type checking
mypy src/
```

### Visualise the graph

```bash
python scripts/visualise_graph.py
# → Outputs a PNG/Mermaid diagram of the LangGraph workflow
```

### Add a new node

1. Implement the node function in `src/graph/nodes.py` with signature `(state: DiscoveryState) -> Dict[str, Any]`
2. Add routing logic to `src/graph/edges.py`
3. Register the node and edge in `src/graph/graph.py`
4. Update `DiscoveryState` in `src/graph/state.py` if new state keys are needed

### Add a new API endpoint

1. Add the route handler to `src/api/routes.py`
2. Add request/response schemas to `src/api/schemas.py`
3. Apply `Depends(require_api_key)` for authenticated routes

---

## Observability

Structured JSON logs are emitted via **structlog** on every significant event (node entry/exit, search results, errors). Example log line:

```json
{"event": "qdrant_hybrid_search_complete", "results_count": 5, "timestamp": "..."}
```

**LangSmith tracing** is enabled when `LANGSMITH_API_KEY` is set. All LangGraph steps, LLM calls, and retrieval operations are traced end-to-end.

Configure in `.env`:
```env
LANGSMITH_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGSMITH_PROJECT=lex-discovery
```

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
<sub>Built with ⚖️ for legal professionals. Not a substitute for qualified legal counsel.</sub>
</div>
