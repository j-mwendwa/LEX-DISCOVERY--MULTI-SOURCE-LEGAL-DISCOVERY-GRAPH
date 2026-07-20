"""
chainlit_app.py — ChatGPT-style Chainlit UI for LEX-DISCOVERY.

Features:
  • ChatGPT-style message streaming with step-by-step pipeline progress
  • PDF lease file upload → LlamaIndex extraction
  • Hybrid search progress (Qdrant/bm25 + bge-small-en-v1.5 dense)
  • Human-in-the-Loop approval via inline action buttons
  • LangSmith observability trace links in the UI
  • Thread persistence via LangGraph AsyncSqliteSaver

Run with:
  chainlit run chainlit_app.py --port 8080
"""
from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import chainlit as cl
from langchain_core.messages import HumanMessage

# ── Startup: configure logging, LlamaIndex, LangSmith ────────────────────────
from src.config import cfg, settings
from src.core.logging import setup_logging
from src.core.tracing import setup_langsmith

setup_logging(app_env=settings.app_env)
setup_langsmith(
    api_key=settings.langsmith_api_key,
    project=settings.langsmith_project,
    tags=cfg.get("langsmith", {}).get("tags", ["legal", "chainlit"]),
)

_UPLOAD_DIR = Path("data") / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session Helpers
# ─────────────────────────────────────────────────────────────────────────────
async def _get_graph_app():
    """Return the compiled LangGraph app (async singleton)."""
    from src.graph.graph import get_app_async
    return await get_app_async()


def _fmt_gaps(gaps: List[str]) -> str:
    if not gaps:
        return "_No compliance gaps identified._"
    return "\n".join(f"• {g}" for g in gaps)


def _fmt_precedents(results: List[Dict]) -> str:
    if not results:
        return "_No precedents found._"
    lines = []
    for r in results:
        lines.append(
            f"**{r.get('title', 'Unknown')}** `{r.get('citation', '')}` "
            f"— score `{r.get('relevance_score', 0):.2f}`\n"
            f"> {r.get('summary', '')[:180]}..."
        )
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# @on_chat_start — Welcome screen and session init
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    """Initialise a new discovery session."""
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("awaiting_hitl", False)
    cl.user_session.set("file_path", None)

    # Compile graph (warm up)
    try:
        app = await _get_graph_app()
        cl.user_session.set("graph_app", app)
    except Exception as exc:
        await cl.Message(
            content=f"⚠️ **Graph initialisation error:** `{exc}`\nPlease check your API keys in `.env`.",
            author="System",
        ).send()
        return

    # LangSmith trace link
    ls_link = ""
    if settings.langsmith_api_key:
        ls_link = (
            f"\n\n🔭 **LangSmith:** [View traces](https://smith.langchain.com/o/"
            f"projects?project={settings.langsmith_project})"
        )

    await cl.Message(
        content=(
            "## ⚖️ LEX-DISCOVERY — Legal AI Assistant\n\n"
            "I'm your AI Lead Attorney. I can:\n"
            "- 📄 Analyse lease PDFs and extract the legal timeline\n"
            "- 🔍 Search Kenyan case law via **Qdrant Cloud hybrid search** (BM25 + semantic)\n"
            "- 🔗 Identify compliance gaps between your facts and established precedents\n"
            "- 📋 Generate a professional legal verdict memo\n\n"
            "**To start:** describe your legal matter below, or upload a lease PDF first.\n"
            f"**Model:** `{settings.hf_model_id}` (Saul Legal AI){ls_link}"
        ),
        author="LEX-DISCOVERY",
    ).send()

    # Offer file upload
    await cl.Message(
        content="📎 **Optional:** Upload your client's lease PDF (or skip and describe the matter).",
        author="LEX-DISCOVERY",
        actions=[
            cl.Action(
                name="upload_pdf",
                label="📄 Upload Lease PDF",
                value="upload",
                description="Upload a PDF lease document for analysis",
            )
        ],
    ).send()


# ─────────────────────────────────────────────────────────────────────────────
# @action_callback — Upload button
# ─────────────────────────────────────────────────────────────────────────────
@cl.action_callback("upload_pdf")
async def handle_upload_action(action: cl.Action):
    files = await cl.AskFileMessage(
        content="Please upload the lease PDF document:",
        accept=["application/pdf", "text/plain"],
        max_size_mb=10,
        timeout=120,
    ).send()

    if files:
        f = files[0]
        dest = _UPLOAD_DIR / f.name
        dest.write_bytes(Path(f.path).read_bytes())
        cl.user_session.set("file_path", str(dest))
        await cl.Message(
            content=f"✅ **Uploaded:** `{f.name}` ({len(Path(f.path).read_bytes()) // 1024} KB)\n\nNow describe your legal matter to begin the discovery pipeline.",
            author="LEX-DISCOVERY",
        ).send()


# ─────────────────────────────────────────────────────────────────────────────
# @on_message — Main pipeline handler
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages and run the discovery pipeline."""

    # ── HITL approval path ───────────────────────────────────────────────────
    if cl.user_session.get("awaiting_hitl"):
        await _handle_hitl_response(message.content)
        return

    # ── Handle file uploads in message ───────────────────────────────────────
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                dest = _UPLOAD_DIR / element.name
                dest.write_bytes(Path(element.path).read_bytes())
                cl.user_session.set("file_path", str(dest))
                await cl.Message(
                    content=f"📄 Lease PDF saved: `{element.name}`",
                    author="LEX-DISCOVERY",
                ).send()

    # ── Run discovery pipeline ───────────────────────────────────────────────
    await _run_discovery_pipeline(matter=message.content)


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline runner
# ─────────────────────────────────────────────────────────────────────────────
async def _run_discovery_pipeline(matter: str):
    """Orchestrate the full 5-stage pipeline with live step updates."""

    thread_id: str = cl.user_session.get("thread_id")
    file_path: str = cl.user_session.get("file_path") or "data/uploads/lease.pdf"
    app = cl.user_session.get("graph_app")

    if not app:
        await cl.Message(content="❌ Graph not initialised. Please refresh.", author="System").send()
        return

    config = {"configurable": {"thread_id": thread_id}}
    initial_state: Dict[str, Any] = {
        "messages": [HumanMessage(content=matter)],
        "hypothesis": "",
        "client_data": None,
        "case_law_results": [],
        "compliance_gaps": [],
        "verdict_approved": False,
        "next_node": "",
        "iteration": 0,
        "file_path": file_path,
    }

    # ── Stage indicators ──────────────────────────────────────────────────────
    stage_map = {
        "lead_attorney_ingestion": ("🧑‍⚖️", "Lead Attorney — Formulating hypothesis"),
        "client_files_runner":     ("📄", "Client Files — Extracting lease data"),
        "case_law_runner":         ("🔍", "Case Law — Hybrid search (BM25 + semantic)"),
        "cross_context_mapping":   ("🔗", "Cross-Context — Identifying compliance gaps"),
        "human_review":            ("🛑", "Human Review — Awaiting Lead Counsel decision"),
        "generate_verdict":        ("📋", "Generating final legal verdict"),
        "rejection":               ("❌", "Pipeline rejected"),
    }

    final_state: Dict[str, Any] = {}
    active_step: Optional[cl.Step] = None
    step_history: List[str] = []

    try:
        async with cl.Step(name="🚀 LEX-DISCOVERY Pipeline", type="run", show_input=True) as root_step:
            root_step.input = matter

            async for event in app.astream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if not isinstance(node_output, dict):
                        continue

                    # Update final state
                    final_state.update(node_output)

                    # Get stage metadata
                    icon, label = stage_map.get(node_name, ("⚙️", node_name))

                    # Close previous step
                    if active_step:
                        active_step.output = f"✅ Complete"
                        await active_step.__aexit__(None, None, None)

                    # Open new step
                    active_step = cl.Step(
                        name=f"{icon} {label}",
                        type="tool",
                        parent_id=root_step.id,
                    )
                    await active_step.__aenter__()

                    # Stage-specific output
                    step_output = await _render_stage_output(node_name, node_output, active_step)
                    step_history.append(f"{icon} **{label}**")

                    # Stop streaming at HITL
                    if node_name == "human_review":
                        if active_step:
                            active_step.output = "⏸️ Awaiting Lead Counsel decision..."
                            await active_step.__aexit__(None, None, None)
                            active_step = None
                        root_step.output = "⏸️ Pipeline paused for human review"
                        break

            # Close final step
            if active_step:
                active_step.output = "✅ Complete"
                await active_step.__aexit__(None, None, None)

    except Exception as exc:
        await cl.Message(
            content=f"❌ **Pipeline error:** `{exc}`",
            author="System",
        ).send()
        return

    # ── Post-pipeline: render summary or HITL prompt ─────────────────────────
    await _post_pipeline_message(final_state, config, app, thread_id)


# ─────────────────────────────────────────────────────────────────────────────
# Stage output renderer
# ─────────────────────────────────────────────────────────────────────────────
async def _render_stage_output(
    node_name: str,
    output: Dict[str, Any],
    step: cl.Step,
) -> str:
    """Render rich output for each pipeline stage into the active step."""

    if node_name == "lead_attorney_ingestion":
        hypothesis = output.get("hypothesis", "")
        step.output = f"**Hypothesis:**\n\n{hypothesis}"

        # Stream hypothesis char by char for ChatGPT feel
        msg = cl.Message(content="", author="🧑‍⚖️ Lead Attorney")
        await msg.send()
        for chunk in _chunk_text(f"**Legal Hypothesis:**\n\n{hypothesis}"):
            await msg.stream_token(chunk)
        await msg.update()
        return hypothesis

    elif node_name == "client_files_runner":
        cd = output.get("client_data") or {}
        meta = cd.get("metadata", {})
        timeline = cd.get("timeline", [])
        clauses = cd.get("clauses", [])

        summary = (
            f"**Tenant:** {meta.get('tenant', 'N/A')}\n"
            f"**Landlord:** {meta.get('landlord', 'N/A')}\n"
            f"**Property:** {meta.get('property_address', 'N/A')}\n"
            f"**Lease:** {meta.get('lease_start', '?')} → {meta.get('lease_end', '?')}\n\n"
            f"**Timeline ({len(timeline)} events):**\n"
            + "\n".join(f"• `{ev.get('date', '?')}` — {ev.get('event', '')}" for ev in timeline)
            + f"\n\n**Notice Clauses ({len(clauses)}):**\n"
            + "\n".join(f"> {c}" for c in clauses)
        )
        step.output = summary

        await cl.Message(
            content=f"📄 **Client File Extraction Complete**\n\n{summary}",
            author="📄 Client Files Agent",
        ).send()
        return summary

    elif node_name == "case_law_runner":
        results = output.get("case_law_results", [])
        formatted = _fmt_precedents(results)
        step.output = f"Found {len(results)} precedents via Qdrant hybrid search"

        await cl.Message(
            content=(
                f"🔍 **Case Law Search Complete — {len(results)} precedents found**\n"
                f"_Using Qdrant Cloud hybrid search: `BAAI/bge-small-en-v1.5` dense + `Qdrant/bm25` sparse (RRF)_\n\n"
                + formatted
            ),
            author="🔍 Case Law Agent",
        ).send()
        return formatted

    elif node_name == "cross_context_mapping":
        gaps = output.get("compliance_gaps", [])
        formatted = _fmt_gaps(gaps)
        step.output = f"Identified {len(gaps)} compliance gap(s)"

        await cl.Message(
            content=f"🔗 **Compliance Gap Analysis — {len(gaps)} gap(s) found**\n\n{formatted}",
            author="🔗 Cross-Context Analyst",
        ).send()
        return formatted

    elif node_name == "generate_verdict":
        msgs = output.get("messages", [])
        verdict = ""
        for m in reversed(msgs):
            if hasattr(m, "name") and m.name == "lead_attorney_verdict":
                verdict = m.content
                break

        step.output = "Verdict generated"

        # Stream the verdict token by token
        msg = cl.Message(content="", author="🧑‍⚖️ Lead Attorney — Final Verdict")
        await msg.send()
        for chunk in _chunk_text(verdict):
            await msg.stream_token(chunk)
        await msg.update()
        return verdict

    elif node_name == "rejection":
        msgs = output.get("messages", [])
        reason = msgs[-1].content if msgs else "Unknown reason"
        step.output = reason
        await cl.Message(content=f"❌ **Pipeline Rejected**\n\n{reason}", author="System").send()
        return reason

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Post-pipeline: summary card + HITL buttons
# ─────────────────────────────────────────────────────────────────────────────
async def _post_pipeline_message(
    state: Dict[str, Any],
    config: dict,
    app,
    thread_id: str,
):
    """Show HITL approval UI or final completion message."""

    # Fetch the persisted snapshot to check actual graph status
    try:
        snapshot = await app.aget_state(config)
        current = snapshot.values if snapshot else state
        next_nodes = snapshot.next if snapshot else []
    except Exception:
        current = state
        next_nodes = []

    gaps = current.get("compliance_gaps", [])
    hypothesis = current.get("hypothesis", "")
    results = current.get("case_law_results", [])

    # ── Pipeline already completed (verdict or rejection) ────────────────────
    if "generate_verdict" not in (next_nodes or []) and "human_review" not in (next_nodes or []):
        msgs = current.get("messages", [])
        for m in reversed(msgs):
            if hasattr(m, "name") and m.name == "lead_attorney_verdict":
                # Already shown in streaming above
                return
        # If rejected
        await cl.Message(
            content=(
                "✅ **Discovery session complete.**\n\n"
                f"Thread ID: `{thread_id}`\n"
                f"Compliance gaps identified: **{len(gaps)}**\n"
                f"Case law precedents reviewed: **{len(results)}**"
            ),
            author="LEX-DISCOVERY",
        ).send()
        return

    # ── HITL prompt ──────────────────────────────────────────────────────────
    cl.user_session.set("awaiting_hitl", True)
    cl.user_session.set("hitl_config", config)
    cl.user_session.set("hitl_app", app)

    gaps_md = _fmt_gaps(gaps)
    summary_card = (
        "---\n"
        "## 🛑 Lead Counsel Review Required\n\n"
        f"**Hypothesis:** {hypothesis}\n\n"
        f"**Precedents reviewed:** {len(results)}\n\n"
        f"**Compliance Gaps ({len(gaps)}):**\n{gaps_md}\n\n"
        "---\n"
        "Please review the findings above and **Approve** to generate the final verdict, "
        "or **Reject** to terminate the pipeline."
    )

    await cl.Message(content=summary_card, author="🛑 Human Review").send()

    # Action buttons
    await cl.Message(
        content="**Your decision:**",
        author="LEX-DISCOVERY",
        actions=[
            cl.Action(
                name="hitl_approve",
                label="✅ Approve — Generate Verdict",
                value="approve",
                description="Approve findings and generate the final legal verdict",
            ),
            cl.Action(
                name="hitl_reject",
                label="❌ Reject — Terminate Pipeline",
                value="reject",
                description="Reject findings and terminate the pipeline",
            ),
        ],
    ).send()


# ─────────────────────────────────────────────────────────────────────────────
# HITL action callbacks
# ─────────────────────────────────────────────────────────────────────────────
@cl.action_callback("hitl_approve")
async def handle_approve(action: cl.Action):
    await _resume_with_verdict(approved=True, notes="Approved via Chainlit UI.")


@cl.action_callback("hitl_reject")
async def handle_reject(action: cl.Action):
    await _resume_with_verdict(approved=False, notes="Rejected by Lead Counsel.")


async def _handle_hitl_response(text: str):
    """Handle freeform text HITL response (approve/reject + optional notes)."""
    lower = text.lower().strip()
    approved = any(w in lower for w in ("approve", "yes", "proceed", "confirm", "ok", "generate"))
    await _resume_with_verdict(approved=approved, notes=text)


async def _resume_with_verdict(approved: bool, notes: str = ""):
    """Resume the LangGraph pipeline after HITL decision."""
    cl.user_session.set("awaiting_hitl", False)

    app = cl.user_session.get("hitl_app") or cl.user_session.get("graph_app")
    config = cl.user_session.get("hitl_config") or {
        "configurable": {"thread_id": cl.user_session.get("thread_id")}
    }

    if not app:
        await cl.Message(content="❌ Session expired. Please start a new chat.", author="System").send()
        return

    decision_label = "✅ APPROVED" if approved else "❌ REJECTED"
    await cl.Message(
        content=f"**Lead Counsel Decision:** {decision_label}\n{f'Notes: {notes}' if notes else ''}",
        author="🧑‍⚖️ Lead Counsel",
    ).send()

    if not approved:
        await cl.Message(
            content="❌ **Pipeline terminated by Lead Counsel.**\n\nStart a new session to re-run the analysis.",
            author="LEX-DISCOVERY",
        ).send()
        return

    # Resume the graph
    try:
        await app.aupdate_state(
            config,
            {"verdict_approved": True},
            as_node="human_review",
        )

        verdict_msg = cl.Message(content="", author="🧑‍⚖️ Lead Attorney — Final Verdict")
        await verdict_msg.send()

        async for event in app.astream(None, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "generate_verdict":
                    msgs = node_output.get("messages", [])
                    for m in reversed(msgs):
                        if hasattr(m, "name") and m.name == "lead_attorney_verdict":
                            for chunk in _chunk_text(m.content):
                                await verdict_msg.stream_token(chunk)
                            break

        await verdict_msg.update()

        # LangSmith link
        if settings.langsmith_api_key:
            thread_id = cl.user_session.get("thread_id")
            await cl.Message(
                content=(
                    f"🔭 **Session complete.** Thread: `{thread_id}`\n"
                    f"[View in LangSmith](https://smith.langchain.com/o/projects?project={settings.langsmith_project})"
                ),
                author="LEX-DISCOVERY",
            ).send()

    except Exception as exc:
        await cl.Message(
            content=f"❌ **Error resuming pipeline:** `{exc}`",
            author="System",
        ).send()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _chunk_text(text: str, size: int = 12):
    """Yield text in small chunks for streaming simulation."""
    for i in range(0, len(text), size):
        yield text[i:i + size]
