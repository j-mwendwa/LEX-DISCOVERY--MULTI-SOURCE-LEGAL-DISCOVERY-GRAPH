"""
src/api/schemas.py — Pydantic request/response models for the LEX-DISCOVERY API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Discovery Start
# ─────────────────────────────────────────────────────────────────────────────
class DiscoveryStartRequest(BaseModel):
    """Request body for POST /discovery/start."""

    client_matter: str = Field(
        ...,
        description="Description of the client's legal matter / dispute.",
        min_length=10,
        examples=["Tenant disputes 18-day eviction notice. Lease requires 30 days."],
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Absolute or relative path to the client's PDF lease file on the server.",
        examples=["data/uploads/lease_john_doe.pdf"],
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional thread ID to continue an existing discovery session.",
    )


class DiscoveryStartResponse(BaseModel):
    """Response body for POST /discovery/start."""

    thread_id: str = Field(..., description="Unique ID for this discovery session.")
    status: str = Field(..., description="Current pipeline status.")
    hypothesis: Optional[str] = Field(
        default=None, description="Formulated legal hypothesis."
    )
    message: str = Field(default="", description="Human-readable status message.")


# ─────────────────────────────────────────────────────────────────────────────
# Discovery Status
# ─────────────────────────────────────────────────────────────────────────────
class TimelineEvent(BaseModel):
    event: str
    date: str


class CaseLawResult(BaseModel):
    title: str
    citation: str
    summary: str
    relevance_score: float


class DiscoveryStatusResponse(BaseModel):
    """Response body for GET /discovery/{thread_id}."""

    thread_id: str
    status: str = Field(
        ...,
        description=(
            "One of: RUNNING, AWAITING_REVIEW, APPROVED, REJECTED, COMPLETE, ERROR"
        ),
    )
    hypothesis: Optional[str] = None
    client_metadata: Optional[Dict[str, str]] = None
    timeline: Optional[List[TimelineEvent]] = None
    case_law_results: Optional[List[CaseLawResult]] = None
    compliance_gaps: Optional[List[str]] = None
    verdict_approved: Optional[bool] = None
    final_verdict: Optional[str] = None
    messages_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Human Approval
# ─────────────────────────────────────────────────────────────────────────────
class ApprovalRequest(BaseModel):
    """Request body for POST /discovery/{thread_id}/approve."""

    verdict_approved: bool = Field(
        ..., description="True to approve and generate final verdict; False to reject."
    )
    counsel_notes: Optional[str] = Field(
        default=None,
        description="Optional notes from lead counsel to include in the final verdict.",
    )


class ApprovalResponse(BaseModel):
    """Response body for POST /discovery/{thread_id}/approve."""

    thread_id: str
    verdict_approved: bool
    status: str
    final_verdict: Optional[str] = None
    message: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Ingest
# ─────────────────────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    file_path: str = Field(
        ...,
        description="Path to the PDF file or directory to ingest into Qdrant.",
    )
    collection_name: Optional[str] = Field(
        default="case_law_precedents",
        description="Target Qdrant collection.",
    )


class IngestResponse(BaseModel):
    chunks_upserted: int
    collection: str
    message: str = "Ingestion complete."


# ─────────────────────────────────────────────────────────────────────────────
# Health / Version
# ─────────────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    app_env: str = "development"
    qdrant_url: str = ""


class VersionResponse(BaseModel):
    version: str = "1.0.0"
    pipeline: str = "Multi-Source Legal Discovery Graph"
