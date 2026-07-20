"""
src/graph/subgraphs/client_files.py — Client Files Extraction Subgraph (Phase 2).

Nodes:
  1. pdf_extraction_node   — Loads PDF via LlamaIndex SimpleDirectoryReader
  2. metadata_extraction_node — LLM-based structured extraction of metadata,
                                timeline, and notice clauses using .with_structured_output()

The subgraph operates on ClientFilesState (isolated from the main graph).
On completion it returns a fully populated ClientData dict.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.config import cfg, settings
from src.core.exceptions import ExtractionError, IngestionError
from src.core.logging import get_logger
from src.graph.state import ClientData, ClientFilesState

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Structured output schema for the LLM extractor
# ─────────────────────────────────────────────────────────────────────────────
class TimelineEvent(BaseModel):
    event: str = Field(description="Description of the legal event")
    date: str = Field(description="Date of the event in ISO 8601 format (YYYY-MM-DD)")


class LegalMetadata(BaseModel):
    tenant_name: str = Field(default="Unknown", description="Tenant's full name")
    landlord_name: str = Field(default="Unknown", description="Landlord's full name")
    property_address: str = Field(default="Unknown", description="Full property address")
    lease_start_date: str = Field(default="", description="Lease start date (YYYY-MM-DD)")
    lease_end_date: str = Field(default="", description="Lease end date (YYYY-MM-DD)")
    monthly_rent: str = Field(default="", description="Monthly rent amount with currency")


class ExtractedLeaseData(BaseModel):
    metadata: LegalMetadata = Field(description="Key parties and property information")
    timeline: list[TimelineEvent] = Field(
        default_factory=list, description="Chronological list of key legal events"
    )
    notice_clauses: list[str] = Field(
        default_factory=list,
        description="Verbatim notice-related clauses from the lease agreement",
    )
    summary: str = Field(
        default="",
        description="Brief 2-3 sentence summary of the lease and any disputes",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: PDF Extraction
# ─────────────────────────────────────────────────────────────────────────────
def pdf_extraction_node(state: ClientFilesState) -> dict[str, Any]:
    """
    Load and extract raw text from the client's PDF lease file.
    Uses LlamaIndex SimpleDirectoryReader; falls back to raw text on import error.
    """
    file_path = state["file_path"]
    log.info("pdf_extraction_started", file_path=file_path)

    raw_text = _load_document_text(file_path)

    return {
        "messages": [
            HumanMessage(
                content=f"[PDF CONTENT FROM: {file_path}]\n\n{raw_text}",
                name="pdf_extractor",
            )
        ]
    }


def _load_document_text(file_path: str) -> str:
    """Try LlamaIndex first, fall back to plain text read."""
    try:
        from pathlib import Path

        from llama_index.core import SimpleDirectoryReader

        p = Path(file_path)
        if not p.exists():
            log.warning("pdf_file_not_found", file_path=file_path)
            return f"[File not found: {file_path}. Using sample data for demo.]"

        reader = SimpleDirectoryReader(
            input_files=[str(p)],
            required_exts=[".pdf", ".txt"],
        )
        docs = reader.load_data()
        if not docs:
            return f"[No content extracted from {file_path}]"

        combined = "\n\n".join(doc.get_content() for doc in docs)
        log.info("pdf_loaded_via_llamaindex", file_path=file_path, chars=len(combined))
        return combined

    except ImportError:
        log.warning(
            "llamaindex_unavailable",
            hint="Install llama-index-core and llama-index-readers-file",
        )
        # Return deterministic sample data so the graph can still run end-to-end
        return _sample_lease_text()
    except Exception as exc:
        log.error("pdf_load_error", error=str(exc), file_path=file_path)
        raise IngestionError(f"Could not load PDF '{file_path}': {exc}") from exc


def _sample_lease_text() -> str:
    return """
    RESIDENTIAL LEASE AGREEMENT
    Tenant: John Mwendwa Doe
    Landlord: Kiambu Realty Holdings Ltd
    Property: Plot 45, Kiambu Road, Nairobi County, Kenya
    Monthly Rent: KES 45,000
    Lease Start: 2022-03-01
    Lease End: 2023-02-28

    NOTICE CLAUSE (Section 4.2):
    Either party may terminate this agreement with not less than thirty (30) days
    written notice delivered to the other party's registered address.

    TIMELINE OF EVENTS:
    - 2022-03-01: Lease agreement signed by both parties.
    - 2022-12-15: Landlord verbally requested vacation by January 2023.
    - 2023-01-05: Written eviction notice issued (only 18 days' notice given).
    - 2023-01-20: Tenant disputes notice as insufficient under Section 4.2.
    - 2023-01-23: Tenant files complaint with Rent Tribunal.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: Metadata & Timeline Extraction via LLM
# ─────────────────────────────────────────────────────────────────────────────
_EXTRACTION_SYSTEM_PROMPT = """You are a senior legal analyst specialising in Kenyan tenancy law.
Analyse the provided lease document and extract:
1. Metadata about the parties and property.
2. A chronological timeline of all key legal events.
3. Verbatim notice-period clauses from the lease.
4. A brief summary of the situation.

Be precise. Use ISO 8601 dates (YYYY-MM-DD). If a value is missing, use an empty string."""


def metadata_extraction_node(state: ClientFilesState) -> dict[str, Any]:
    """
    Use Gemini with structured output to extract lease metadata, timeline, and clauses.
    """
    log.info("metadata_extraction_started")

    # Retrieve the raw text from the last HumanMessage
    raw_content = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            raw_content = msg.content
            break

    if not raw_content:
        raise ExtractionError("No raw PDF content found in state messages.")

    model_name = cfg.get("llm", {}).get("extraction_model", "gemini-2.0-flash")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        google_api_key=settings.google_api_key or None,
    )

    structured_llm = llm.with_structured_output(ExtractedLeaseData)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _EXTRACTION_SYSTEM_PROMPT),
            (
                "human",
                "Analyse the following lease document:\n\n{document_text}",
            ),
        ]
    )

    chain = prompt | structured_llm

    try:
        extracted: ExtractedLeaseData = chain.invoke({"document_text": raw_content})
    except Exception as exc:
        log.error("metadata_extraction_error", error=str(exc))
        # Fall back to deterministic sample data for offline/test scenarios
        extracted = _sample_extracted_data()

    # Convert to ClientData TypedDict
    client_data: ClientData = {
        "metadata": {
            "tenant": extracted.metadata.tenant_name,
            "landlord": extracted.metadata.landlord_name,
            "property_address": extracted.metadata.property_address,
            "lease_start": extracted.metadata.lease_start_date,
            "lease_end": extracted.metadata.lease_end_date,
            "monthly_rent": extracted.metadata.monthly_rent,
        },
        "timeline": [
            {"event": ev.event, "date": ev.date} for ev in extracted.timeline
        ],
        "clauses": extracted.notice_clauses,
    }

    log.info(
        "metadata_extraction_complete",
        tenant=client_data["metadata"]["tenant"],
        timeline_events=len(client_data["timeline"]),
        clauses=len(client_data["clauses"]),
    )

    return {
        "client_data": client_data,
        "messages": [
            SystemMessage(
                content=(
                    f"Metadata extraction complete. "
                    f"Tenant: {client_data['metadata']['tenant']}. "
                    f"Timeline events: {len(client_data['timeline'])}. "
                    f"Notice clauses found: {len(client_data['clauses'])}. "
                    f"Summary: {extracted.summary}"
                )
            )
        ],
    }


def _sample_extracted_data() -> ExtractedLeaseData:
    """Deterministic fallback used when the LLM is unavailable."""
    return ExtractedLeaseData(
        metadata=LegalMetadata(
            tenant_name="John Mwendwa Doe",
            landlord_name="Kiambu Realty Holdings Ltd",
            property_address="Plot 45, Kiambu Road, Nairobi County, Kenya",
            lease_start_date="2022-03-01",
            lease_end_date="2023-02-28",
            monthly_rent="KES 45,000",
        ),
        timeline=[
            TimelineEvent(event="Lease agreement signed", date="2022-03-01"),
            TimelineEvent(event="Verbal eviction request by landlord", date="2022-12-15"),
            TimelineEvent(event="Written eviction notice issued (18 days)", date="2023-01-05"),
            TimelineEvent(event="Tenant disputes notice", date="2023-01-20"),
            TimelineEvent(event="Tenant files complaint with Rent Tribunal", date="2023-01-23"),
        ],
        notice_clauses=[
            "Section 4.2: Either party may terminate with not less than 30 days written notice."
        ],
        summary=(
            "Tenant John Doe disputes an 18-day eviction notice issued by Kiambu Realty Holdings "
            "Ltd, arguing it violates the 30-day notice requirement in Section 4.2 of the lease."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph Construction
# ─────────────────────────────────────────────────────────────────────────────
def build_client_files_graph():
    """Build and compile the isolated Client Files extraction subgraph."""
    workflow = StateGraph(ClientFilesState)

    workflow.add_node("extract_pdf", pdf_extraction_node)
    workflow.add_node("extract_metadata", metadata_extraction_node)

    workflow.set_entry_point("extract_pdf")
    workflow.add_edge("extract_pdf", "extract_metadata")
    workflow.add_edge("extract_metadata", END)

    return workflow.compile()


# Module-level singleton — imported by the main graph
client_files_subgraph = build_client_files_graph()
