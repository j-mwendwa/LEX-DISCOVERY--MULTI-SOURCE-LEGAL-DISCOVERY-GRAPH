"""
src/graph/edges.py — Routing logic for the Lead Attorney main graph.

Provides:
  route_after_ingestion()   — routes to client_files or rejection after supervisor node.
  route_after_cross_context() — routes to human_review after mapping.
  route_after_human_review()  — routes to generate_verdict or rejection.
  route_after_verdict()       — routes to END.
"""
from __future__ import annotations

from typing import Literal

from src.core.logging import get_logger
from src.graph.state import DiscoveryState

log = get_logger(__name__)


def route_after_ingestion(
    state: DiscoveryState,
) -> Literal["client_files_runner", "rejection"]:
    """Route after the lead_attorney_ingestion node."""
    next_node = state.get("next_node", "client_files")
    if next_node == "rejection":
        log.info("routing_to_rejection", from_node="lead_attorney_ingestion")
        return "rejection"
    return "client_files_runner"


def route_after_client_files(
    state: DiscoveryState,
) -> Literal["case_law_runner", "rejection"]:
    """Route after the client_files_runner node."""
    next_node = state.get("next_node", "case_law")
    if next_node == "rejection":
        log.info("routing_to_rejection", from_node="client_files_runner")
        return "rejection"
    return "case_law_runner"


def route_after_case_law(
    state: DiscoveryState,
) -> Literal["cross_context_mapping", "rejection"]:
    """Route after the case_law_runner node."""
    next_node = state.get("next_node", "cross_context")
    if next_node == "rejection":
        return "rejection"
    return "cross_context_mapping"


def route_after_human_review(
    state: DiscoveryState,
) -> Literal["generate_verdict", "rejection"]:
    """Route after the human_review HITL node."""
    approved = state.get("verdict_approved", False)
    if approved:
        log.info("routing_to_verdict_generation", approved=True)
        return "generate_verdict"
    log.info("routing_to_rejection", approved=False)
    return "rejection"
