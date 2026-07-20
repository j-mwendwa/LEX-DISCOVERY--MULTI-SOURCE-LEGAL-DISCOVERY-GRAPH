"""
tests/unit/test_state.py — Unit tests for DiscoveryState and subgraph states.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import CaseLawState, ClientFilesState, DiscoveryState


def test_discovery_state_has_required_fields():
    """DiscoveryState TypedDict has all required fields."""
    state: DiscoveryState = {
        "messages": [],
        "hypothesis": "Test hypothesis",
        "client_data": None,
        "case_law_results": [],
        "compliance_gaps": [],
        "verdict_approved": False,
        "next_node": "",
        "iteration": 0,
    }
    assert state["hypothesis"] == "Test hypothesis"
    assert state["verdict_approved"] is False
    assert isinstance(state["case_law_results"], list)


def test_client_files_state_has_required_fields():
    """ClientFilesState TypedDict has all required fields."""
    state: ClientFilesState = {
        "file_path": "data/sample.pdf",
        "client_data": None,
        "messages": [HumanMessage(content="test")],
    }
    assert state["file_path"] == "data/sample.pdf"
    assert state["client_data"] is None


def test_case_law_state_has_required_fields():
    """CaseLawState TypedDict has all required fields."""
    state: CaseLawState = {
        "query": "eviction notice Kenya",
        "results": [],
        "messages": [],
    }
    assert state["query"] == "eviction notice Kenya"
    assert isinstance(state["results"], list)


def test_discovery_state_messages_accept_base_messages():
    """messages field must accept BaseMessage instances."""
    state: DiscoveryState = {
        "messages": [
            HumanMessage(content="Client dispute"),
            AIMessage(content="Hypothesis formulated."),
        ],
        "hypothesis": "",
        "client_data": None,
        "case_law_results": [],
        "compliance_gaps": [],
        "verdict_approved": False,
        "next_node": "",
        "iteration": 0,
    }
    assert len(state["messages"]) == 2
