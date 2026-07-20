from typing import TypedDict, Annotated, List, Dict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ClientData(TypedDict):
    metadata: Dict[str, str]
    timeline: List[Dict[str, str]]
    clauses: List[str]

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
    client_data: Optional[ClientData]

    # Results from precedent search (Phase 3)
    case_law_results: Annotated[List[CaseLawResult], list]

    # Analysis result: gaps between client timeline and precedent (Phase 4)
    compliance_gaps: Annotated[List[str], list]

    # Human approval flag (Phase 5)
    verdict_approved: bool

    # Standard conversation history
    messages: Annotated[List[BaseMessage], add_messages]

    # Internal routing/loop control
    next_node: Optional[str]
    iteration: int

class ClientFilesState(TypedDict):
    """
    State for the isolated Client Files subgraph.
    """
    file_path: str
    client_data: Optional[ClientData]
    messages: Annotated[List[BaseMessage], add_messages]

class CaseLawState(TypedDict):
    """
    State for the isolated Case Law subgraph.
    """
    query: str
    results: List[CaseLawResult]
    messages: Annotated[List[BaseMessage], add_messages]
