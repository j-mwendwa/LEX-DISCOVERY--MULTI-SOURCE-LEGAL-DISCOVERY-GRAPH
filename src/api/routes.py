"""
src/api/routes.py — All FastAPI route handlers for LEX-DISCOVERY.

Endpoints:
  GET  /health                        — Liveness probe
  GET  /version                       — Pipeline version
  POST /discovery/start               — Initiate discovery pipeline
  GET  /discovery/{thread_id}         — Poll discovery status
  POST /discovery/{thread_id}/approve — Submit human verdict (HITL resume)
  POST /ingest                        — Ingest case law PDFs into Qdrant
  POST /ingest/upload                 — Upload + ingest PDF files
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from langchain_core.messages import HumanMessage

from src.api.auth import require_api_key
from src.api.schemas import (
    ApprovalRequest,
    ApprovalResponse,
    CaseLawResult,
    DiscoveryStartRequest,
    DiscoveryStartResponse,
    DiscoveryStatusResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    TimelineEvent,
    VersionResponse,
)
from src.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

_UPLOAD_DIR = Path("data") / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Health & Version
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Liveness probe — no auth required."""
    return HealthResponse(
        status="ok",
        app_env=settings.app_env,
        qdrant_url=settings.qdrant_url,
    )


@router.get("/version", response_model=VersionResponse, tags=["System"])
async def version() -> VersionResponse:
    """Pipeline version — no auth required."""
    return VersionResponse(version="1.0.0", pipeline="Multi-Source Legal Discovery Graph")


# ─────────────────────────────────────────────────────────────────────────────
# POST /discovery/start
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/discovery/start",
    response_model=DiscoveryStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Discovery"],
    dependencies=[Depends(require_api_key)],
)
async def start_discovery(
    request_body: DiscoveryStartRequest,
    request: Request,
) -> DiscoveryStartResponse:
    """
    Start the Multi-Source Legal Discovery pipeline.

    Runs the graph up to the HITL interrupt (human_review node), then pauses.
    Returns the thread_id for status polling and approval.
    """
    thread_id = request_body.thread_id or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(thread_id=thread_id)
    log.info("discovery_start_requested", thread_id=thread_id)

    app = request.app.state.graph_app
    if app is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph engine not initialised. Check startup logs.",
        )

    config = {"configurable": {"thread_id": thread_id}}
    initial_state: dict[str, Any] = {
        "messages": [HumanMessage(content=request_body.client_matter)],
        "hypothesis": "",
        "client_data": None,
        "case_law_results": [],
        "compliance_gaps": [],
        "verdict_approved": False,
        "next_node": "",
        "iteration": 0,
        "file_path": request_body.file_path or "data/uploads/lease.pdf",
    }

    try:
        # Stream events until the graph pauses at the HITL interrupt
        result_state: dict[str, Any] | None = None
        async for event in app.astream(initial_state, config=config):
            log.debug("graph_event", keys=list(event.keys()))
            # The last non-interrupt event holds the state
            for _node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    result_state = node_output

        # Fetch the persisted state snapshot
        snapshot = await app.aget_state(config)
        current_state = snapshot.values if snapshot else (result_state or {})

        hypothesis = current_state.get("hypothesis", "")
        pipeline_status = _infer_status(current_state)

        log.info(
            "discovery_pipeline_paused",
            thread_id=thread_id,
            status=pipeline_status,
        )

        return DiscoveryStartResponse(
            thread_id=thread_id,
            status=pipeline_status,
            hypothesis=hypothesis,
            message=(
                "Pipeline is paused at human review. "
                "Call POST /discovery/{thread_id}/approve to submit your decision."
                if pipeline_status == "AWAITING_REVIEW"
                else "Pipeline completed or rejected."
            ),
        )

    except Exception as exc:
        log.error("discovery_start_error", thread_id=thread_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# GET /discovery/{thread_id}
# ─────────────────────────────────────────────────────────────────────────────
@router.get(
    "/discovery/{thread_id}",
    response_model=DiscoveryStatusResponse,
    tags=["Discovery"],
    dependencies=[Depends(require_api_key)],
)
async def get_discovery_status(
    thread_id: str,
    request: Request,
) -> DiscoveryStatusResponse:
    """Poll the status of an ongoing or completed discovery session."""
    structlog.contextvars.bind_contextvars(thread_id=thread_id)
    app = request.app.state.graph_app

    if app is None:
        raise HTTPException(status_code=503, detail="Graph engine not initialised.")

    try:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await app.aget_state(config)
        if snapshot is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No discovery session found for thread_id: {thread_id}",
            )

        state = snapshot.values
        client_data = state.get("client_data") or {}
        meta = client_data.get("metadata", {})
        timeline_raw = client_data.get("timeline", [])
        case_law_raw = state.get("case_law_results", [])
        messages = state.get("messages", [])

        # Extract final verdict from last AI message if complete
        final_verdict: str | None = None
        for msg in reversed(messages):
            if hasattr(msg, "name") and msg.name == "lead_attorney_verdict":
                final_verdict = msg.content
                break

        return DiscoveryStatusResponse(
            thread_id=thread_id,
            status=_infer_status(state),
            hypothesis=state.get("hypothesis"),
            client_metadata=meta if meta else None,
            timeline=[TimelineEvent(**ev) for ev in timeline_raw] if timeline_raw else None,
            case_law_results=[CaseLawResult(**r) for r in case_law_raw] if case_law_raw else None,
            compliance_gaps=state.get("compliance_gaps"),
            verdict_approved=state.get("verdict_approved"),
            final_verdict=final_verdict,
            messages_count=len(messages),
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error("status_check_error", thread_id=thread_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# POST /discovery/{thread_id}/approve
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/discovery/{thread_id}/approve",
    response_model=ApprovalResponse,
    tags=["Discovery"],
    dependencies=[Depends(require_api_key)],
)
async def approve_discovery(
    thread_id: str,
    approval: ApprovalRequest,
    request: Request,
) -> ApprovalResponse:
    """
    Resume the paused HITL node with lead counsel's decision.

    This resumes the LangGraph interrupt in human_review, passing the
    approval decision. If approved, the graph continues to generate_verdict.
    """
    structlog.contextvars.bind_contextvars(thread_id=thread_id)
    log.info(
        "hitl_approval_received",
        thread_id=thread_id,
        approved=approval.verdict_approved,
    )

    app = request.app.state.graph_app
    if app is None:
        raise HTTPException(status_code=503, detail="Graph engine not initialised.")

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Check that the graph is indeed paused at human_review
        snapshot = await app.aget_state(config)
        if snapshot is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No session found for thread_id: {thread_id}",
            )

        next_nodes = snapshot.next
        if "human_review" not in (next_nodes or []):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Session '{thread_id}' is not awaiting human review. "
                    f"Current next node(s): {next_nodes}"
                ),
            )

        async for event in app.astream(
            # Pass None to resume from interrupt; provide interrupt value via update_state
            None,
            config=config,
            stream_mode="updates",
        ):
            for _node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    pass  # consume stream events

        # Inject the human verdict and resume
        await app.aupdate_state(
            config,
            {"verdict_approved": approval.verdict_approved},
            as_node="human_review",
        )

        # Re-stream to completion after state update
        async for event in app.astream(None, config=config):
            for _node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    pass  # consume stream events

        # Get final verdict from snapshot
        final_snapshot = await app.aget_state(config)
        final_vals = final_snapshot.values if final_snapshot else {}
        messages = final_vals.get("messages", [])
        verdict_text: str | None = None
        for msg in reversed(messages):
            if hasattr(msg, "name") and msg.name == "lead_attorney_verdict":
                verdict_text = msg.content
                break

        pipeline_status = "COMPLETE" if approval.verdict_approved else "REJECTED"

        return ApprovalResponse(
            thread_id=thread_id,
            verdict_approved=approval.verdict_approved,
            status=pipeline_status,
            final_verdict=verdict_text,
            message=(
                "Verdict generated successfully."
                if approval.verdict_approved
                else "Discovery pipeline rejected by lead counsel."
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error("approval_error", thread_id=thread_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# POST /ingest
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Ingestion"],
    dependencies=[Depends(require_api_key)],
)
async def ingest_documents(body: IngestRequest) -> IngestResponse:
    """Ingest case law PDFs from a server-side path into Qdrant."""
    try:
        from src.ingestion.llamaindex_pipeline import ingest_lease_pdf

        chunks = ingest_lease_pdf(
            file_path=body.file_path,
            collection_name=body.collection_name or "case_law_precedents",
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
        )
        return IngestResponse(
            chunks_upserted=chunks,
            collection=body.collection_name or "case_law_precedents",
        )
    except Exception as exc:
        log.error("ingest_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# POST /ingest/upload
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/ingest/upload",
    response_model=IngestResponse,
    tags=["Ingestion"],
    dependencies=[Depends(require_api_key)],
)
async def ingest_upload(
    collection_name: str = "case_law_precedents",
    file: Annotated[UploadFile, File(...)] = ...,
) -> IngestResponse:
    """Upload a PDF then ingest it into Qdrant."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    dest = _UPLOAD_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)
    log.info("file_uploaded", filename=file.filename, bytes=len(content))

    try:
        from src.ingestion.llamaindex_pipeline import ingest_lease_pdf

        chunks = ingest_lease_pdf(
            file_path=str(dest),
            collection_name=collection_name,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            skip_path_validation=True,  # file was just uploaded to data/uploads
        )
        return IngestResponse(
            chunks_upserted=chunks,
            collection=collection_name,
            message=f"Uploaded and ingested '{file.filename}' — {chunks} chunks.",
        )
    except Exception as exc:
        log.error("upload_ingest_error", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _infer_status(state: dict[str, Any]) -> str:
    """Infer a human-readable pipeline status from the current state."""
    if state.get("verdict_approved") and state.get("compliance_gaps"):
        return "COMPLETE"
    if state.get("compliance_gaps") and not state.get("verdict_approved"):
        return "AWAITING_REVIEW"
    if state.get("case_law_results"):
        return "RUNNING"
    if state.get("hypothesis"):
        return "RUNNING"
    return "INITIALISED"
