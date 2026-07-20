"""
src/graph/graph.py — Main Lead Attorney discovery graph (Phase 4).

Builds and compiles the full 5-stage pipeline:
  lead_attorney_ingestion → client_files_runner → case_law_runner
    → cross_context_mapping → human_review → generate_verdict → END
                                           ↘ rejection → END

Persistence: AsyncSqliteSaver (async FastAPI path) / SqliteSaver (sync tests).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.core.logging import get_logger
from src.graph.edges import (
    route_after_case_law,
    route_after_client_files,
    route_after_human_review,
    route_after_ingestion,
)
from src.graph.nodes import (
    case_law_runner,
    client_files_runner,
    cross_context_mapping,
    generate_verdict,
    human_review,
    lead_attorney_ingestion,
    rejection_node,
)
from src.graph.state import DiscoveryState

log = get_logger(__name__)

_DB_PATH = Path("data") / "checkpoints.db"
_async_app = None  # singleton for async FastAPI path


def build_graph(checkpointer=None):
    """
    Construct and compile the Lead Attorney LangGraph workflow.

    Args:
        checkpointer: Optional LangGraph checkpointer (SqliteSaver / AsyncSqliteSaver).

    Returns:
        Compiled LangGraph app.
    """
    workflow = StateGraph(DiscoveryState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("lead_attorney_ingestion", lead_attorney_ingestion)
    workflow.add_node("client_files_runner", client_files_runner)
    workflow.add_node("case_law_runner", case_law_runner)
    workflow.add_node("cross_context_mapping", cross_context_mapping)
    workflow.add_node("human_review", human_review)
    workflow.add_node("generate_verdict", generate_verdict)
    workflow.add_node("rejection", rejection_node)

    # ── Entry point ────────────────────────────────────────────────────────────
    workflow.set_entry_point("lead_attorney_ingestion")

    # ── Conditional edges ─────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "lead_attorney_ingestion",
        route_after_ingestion,
        {
            "client_files_runner": "client_files_runner",
            "rejection": "rejection",
        },
    )

    workflow.add_conditional_edges(
        "client_files_runner",
        route_after_client_files,
        {
            "case_law_runner": "case_law_runner",
            "rejection": "rejection",
        },
    )

    workflow.add_conditional_edges(
        "case_law_runner",
        route_after_case_law,
        {
            "cross_context_mapping": "cross_context_mapping",
            "rejection": "rejection",
        },
    )

    # cross_context → human_review (always, HITL is mandatory)
    workflow.add_edge("cross_context_mapping", "human_review")

    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "generate_verdict": "generate_verdict",
            "rejection": "rejection",
        },
    )

    # ── Terminal edges ─────────────────────────────────────────────────────────
    workflow.add_edge("generate_verdict", END)
    workflow.add_edge("rejection", END)

    log.info("graph_built", nodes=list(workflow.nodes.keys()))

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


def get_app(db_path: str = str(_DB_PATH)):
    """
    Build the graph with a synchronous SqliteSaver checkpointer.
    Suitable for scripts, CLI tools, and tests.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        log.info("sync_graph_compiled", db_path=db_path)
        return app


async def get_app_async(db_path: str = str(_DB_PATH)):
    """
    Build the graph with an AsyncSqliteSaver checkpointer (singleton).
    Used by the FastAPI lifespan. Cached after first call.
    """
    global _async_app
    if _async_app is not None:
        return _async_app

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            _async_app = build_graph(checkpointer=checkpointer)
            log.info("async_graph_compiled", db_path=db_path)
            return _async_app
    except ImportError:
        log.warning(
            "async_sqlite_unavailable",
            hint="Install aiosqlite for async checkpointing",
        )
        _async_app = build_graph()
        return _async_app
