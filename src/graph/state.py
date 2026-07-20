from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ClientData(TypedDict):
    metadata: dict[str, str]
    timeline: list[dict[str, str]]
    clauses: list[str]


class CaseLawResult(TypedDict):
    title: str
    citation: str
    summary: str
    relevance_score: float


class DiscoveryState(TypedDict, total=False):
    """
    Main state for the Multi-Stage Discovery Pipeline.
    """

    # The high-level legal hypothesis from the supervisor
    hypothesis: str

    # Path to the client's PDF lease file (optional; defaults in node)
    file_path: str

    # Data extracted from client files (Phase 2)
    client_data: ClientData | None

    # Results from precedent search (Phase 3)
    case_law_results: Annotated[list[CaseLawResult], list]

    # Analysis result: gaps between client timeline and precedent (Phase 4)
    compliance_gaps: Annotated[list[str], list]

    # Human approval flag (Phase 5)
    verdict_approved: bool

    # Standard conversation history
    messages: Annotated[list[BaseMessage], add_messages]

    # Internal routing/loop control
    next_node: str | None
    iteration: int


class ClientFilesState(TypedDict):
    """
    State for the isolated Client Files subgraph.
    """

    file_path: str
    client_data: ClientData | None
    messages: Annotated[list[BaseMessage], add_messages]


class CaseLawState(TypedDict):
    """
    State for the isolated Case Law subgraph.
    """

    query: str
    results: list[CaseLawResult]
    messages: Annotated[list[BaseMessage], add_messages]
