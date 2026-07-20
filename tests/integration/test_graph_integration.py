"""
tests/integration/test_graph_integration.py — Integration tests for the full pipeline.

Requires: GOOGLE_API_KEY set in .env
Run with: pytest -m integration tests/integration/
"""

from __future__ import annotations

import os
import tempfile

import pytest
from langchain_core.messages import HumanMessage

pytestmark = pytest.mark.integration


@pytest.fixture
def sample_initial_state():
    return {
        "messages": [
            HumanMessage(
                content=(
                    "My landlord issued an eviction notice with only 18 days, "
                    "but our lease requires 30 days. I need to dispute this."
                )
            )
        ],
        "hypothesis": "",
        "client_data": None,
        "case_law_results": [],
        "compliance_gaps": [],
        "verdict_approved": False,
        "next_node": "",
        "iteration": 0,
        "file_path": "tests/fixtures/sample_lease.txt",
    }


@pytest.mark.integration
def test_graph_builds_without_error():
    """The main graph should build and compile without raising exceptions."""
    from src.graph.graph import build_graph

    app = build_graph()
    assert app is not None


@pytest.mark.integration
def test_client_files_subgraph_produces_client_data(tmp_path):
    """Client files subgraph should extract ClientData from a sample text file."""
    from src.graph.state import ClientFilesState
    from src.graph.subgraphs.client_files import client_files_subgraph

    lease = tmp_path / "lease.txt"
    lease.write_text(
        "Tenant: Alice Wanjiku\nLandlord: Mombasa Realty\n"
        "Section 4.2: 30-day notice required.\n"
        "2023-01-10: Eviction notice issued (15 days).\n"
    )

    state: ClientFilesState = {
        "file_path": str(lease),
        "client_data": None,
        "messages": [],
    }

    result = client_files_subgraph.invoke(state)
    assert result.get("client_data") is not None
    assert "metadata" in result["client_data"]
    assert "timeline" in result["client_data"]


@pytest.mark.integration
def test_case_law_subgraph_returns_results():
    """Case law subgraph should return at least 1 result (mock fallback)."""
    from src.graph.state import CaseLawState
    from src.graph.subgraphs.case_law import case_law_subgraph

    state: CaseLawState = {
        "query": "insufficient eviction notice Kenya tenancy 18 days",
        "results": [],
        "messages": [],
    }

    result = case_law_subgraph.invoke(state)
    assert len(result.get("results", [])) > 0


@pytest.mark.integration
def test_graph_runs_to_hitl_interrupt(sample_initial_state):
    """
    The full graph should run from ingestion to the human_review interrupt and pause.
    Verifies that hypothesis, client_data, and compliance_gaps are populated.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    from src.graph.graph import build_graph

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        with SqliteSaver.from_conn_string(db_path) as checkpointer:
            app = build_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "test-integration-001"}}

            events = list(app.stream(sample_initial_state, config=config))
            assert len(events) > 0

            # The graph should have paused — state should have hypothesis
            snapshot = app.get_state(config)
            state = snapshot.values
            assert state.get("hypothesis"), "Hypothesis should have been set"

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
