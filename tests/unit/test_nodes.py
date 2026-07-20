"""
tests/unit/test_nodes.py — Unit tests for graph node functions (mocked LLM).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import SystemMessage

from src.graph.state import DiscoveryState


# ─────────────────────────────────────────────────────────────────────────────
# rejection_node
# ─────────────────────────────────────────────────────────────────────────────
def test_rejection_node_returns_end_routing():
    """rejection_node should set next_node to END and return an AIMessage."""
    from src.graph.nodes import rejection_node

    state: DiscoveryState = {
        "messages": [SystemMessage(content="ERROR: No client matter provided.")],
        "hypothesis": "",
        "client_data": None,
        "case_law_results": [],
        "compliance_gaps": [],
        "verdict_approved": False,
        "next_node": "rejection",
        "iteration": 0,
    }

    result = rejection_node(state)
    assert result["next_node"] == "END"
    assert len(result["messages"]) == 1
    assert "Terminated" in result["messages"][0].content


# ─────────────────────────────────────────────────────────────────────────────
# lead_attorney_ingestion — no matter provided
# ─────────────────────────────────────────────────────────────────────────────
def test_lead_attorney_ingestion_rejects_empty_matter():
    """lead_attorney_ingestion should route to rejection when no matter is given."""
    from src.graph.nodes import lead_attorney_ingestion

    state: DiscoveryState = {
        "messages": [],
        "hypothesis": "",
        "client_data": None,
        "case_law_results": [],
        "compliance_gaps": [],
        "verdict_approved": False,
        "next_node": "",
        "iteration": 0,
    }

    result = lead_attorney_ingestion(state)
    assert result["next_node"] == "rejection"


# ─────────────────────────────────────────────────────────────────────────────
# cross_context_mapping — offline fallback
# ─────────────────────────────────────────────────────────────────────────────
def test_cross_context_mapping_uses_fallback_when_llm_unavailable(
    sample_client_data, sample_case_law_results
):
    """cross_context_mapping should return gaps using the offline fallback."""
    from src.graph.nodes import cross_context_mapping

    state: DiscoveryState = {
        "messages": [],
        "hypothesis": "18-day notice violates 30-day clause.",
        "client_data": sample_client_data,
        "case_law_results": sample_case_law_results,
        "compliance_gaps": [],
        "verdict_approved": False,
        "next_node": "",
        "iteration": 1,
    }

    with patch("src.graph.nodes._get_llm") as mock_llm_factory:
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        # Simulate LLM failure to trigger fallback
        mock_chain.invoke.side_effect = RuntimeError("LLM unavailable")
        mock_llm.__or__ = MagicMock(return_value=mock_chain)
        mock_llm_factory.return_value = mock_llm

        result = cross_context_mapping(state)

    assert len(result["compliance_gaps"]) > 0
    assert result["next_node"] == "human_review"


# ─────────────────────────────────────────────────────────────────────────────
# edges
# ─────────────────────────────────────────────────────────────────────────────
def test_route_after_ingestion_client_files():
    """Should route to client_files_runner when next_node is not rejection."""
    from src.graph.edges import route_after_ingestion

    state: DiscoveryState = {
        "messages": [],
        "hypothesis": "Test",
        "client_data": None,
        "case_law_results": [],
        "compliance_gaps": [],
        "verdict_approved": False,
        "next_node": "client_files",
        "iteration": 1,
    }
    assert route_after_ingestion(state) == "client_files_runner"


def test_route_after_ingestion_rejection():
    """Should route to rejection when next_node is rejection."""
    from src.graph.edges import route_after_ingestion

    state: DiscoveryState = {
        "messages": [],
        "hypothesis": "",
        "client_data": None,
        "case_law_results": [],
        "compliance_gaps": [],
        "verdict_approved": False,
        "next_node": "rejection",
        "iteration": 0,
    }
    assert route_after_ingestion(state) == "rejection"


def test_route_after_human_review_approval():
    """Should route to generate_verdict when verdict_approved is True."""
    from src.graph.edges import route_after_human_review

    state: DiscoveryState = {
        "messages": [],
        "hypothesis": "Test",
        "client_data": None,
        "case_law_results": [],
        "compliance_gaps": [],
        "verdict_approved": True,
        "next_node": "generate_verdict",
        "iteration": 2,
    }
    assert route_after_human_review(state) == "generate_verdict"


def test_route_after_human_review_rejection():
    """Should route to rejection when verdict_approved is False."""
    from src.graph.edges import route_after_human_review

    state: DiscoveryState = {
        "messages": [],
        "hypothesis": "Test",
        "client_data": None,
        "case_law_results": [],
        "compliance_gaps": [],
        "verdict_approved": False,
        "next_node": "rejection",
        "iteration": 2,
    }
    assert route_after_human_review(state) == "rejection"
