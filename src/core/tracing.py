"""
src/core/tracing.py — LangSmith Observability setup for LEX-DISCOVERY.

Provides:
  - setup_langsmith(): Configure LangSmith env vars + verify connectivity
  - get_tracer(): Return a LangChainTracer for inline use in chains
  - get_run_url(): Return the LangSmith run URL for a given run ID
"""
from __future__ import annotations

import os
from typing import Optional

from src.core.logging import get_logger

log = get_logger(__name__)


def setup_langsmith(
    api_key: str = "",
    project: str = "lex-discovery",
    endpoint: str = "https://api.smith.langchain.com",
    tags: Optional[list] = None,
) -> bool:
    """
    Enable LangSmith tracing if an API key is provided.

    Returns:
        True if tracing was enabled, False otherwise.
    """
    if not api_key:
        log.info("langsmith_tracing_disabled", reason="No LANGSMITH_API_KEY set")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project

    # Verify connectivity (non-blocking)
    try:
        from langsmith import Client

        client = Client(api_url=endpoint, api_key=api_key)
        # Lightweight check — list projects
        _ = list(client.list_projects(limit=1))
        log.info(
            "langsmith_tracing_enabled",
            project=project,
            endpoint=endpoint,
            tags=tags or [],
        )
        return True
    except Exception as exc:
        log.warning(
            "langsmith_connectivity_warning",
            error=str(exc),
            note="Tracing env vars are set but connection check failed — will retry on first trace",
        )
        return True  # Env vars are set; traces will go through when connectivity returns


def get_tracer(run_name: str = "lex-discovery"):
    """
    Return a LangChainTracer for passing as a callback to LangChain chains.
    Returns None if LangSmith is not configured.
    """
    if not os.getenv("LANGCHAIN_API_KEY"):
        return None
    try:
        from langsmith import Client
        from langchain.callbacks.tracers import LangChainTracer

        return LangChainTracer(
            project_name=os.getenv("LANGCHAIN_PROJECT", "lex-discovery"),
            client=Client(
                api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
                api_key=os.getenv("LANGCHAIN_API_KEY"),
            ),
        )
    except ImportError:
        log.warning("langsmith_tracer_unavailable", hint="pip install langsmith")
        return None


def get_callbacks(run_name: str = "lex-discovery") -> list:
    """Return a list of callbacks (tracer if configured, empty list otherwise)."""
    tracer = get_tracer(run_name)
    return [tracer] if tracer else []
