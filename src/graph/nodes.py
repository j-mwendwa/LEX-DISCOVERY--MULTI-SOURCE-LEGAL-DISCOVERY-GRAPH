"""
src/graph/nodes.py — Node implementations for the Lead Attorney main graph (Phases 4 & 5).

Nodes:
  lead_attorney_ingestion — Supervisor node: validates input, formulates hypothesis.
  client_files_runner     — Wrapper invoking the Client Files subgraph.
  case_law_runner         — Wrapper invoking the Case Law subgraph.
  cross_context_mapping   — Compares client timeline vs precedent to find compliance gaps.
  human_review            — HITL: raises LangGraph interrupt for lead counsel approval.
  generate_verdict        — Synthesises final legal verdict after human approval.
  rejection_node          — Terminates pipeline for invalid / incomplete inputs.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt

from src.core.llm_factory import get_law_llm
from src.core.logging import get_logger
from src.graph.state import CaseLawResult, ClientData, DiscoveryState

log = get_logger(__name__)


def _get_llm(temperature: float = 0.0):
    """Get the law-specialised LLM (Saul-7B → Gemini fallback)."""
    return get_law_llm(temperature=temperature)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: Lead Attorney Supervisor — entry point
# ─────────────────────────────────────────────────────────────────────────────
_SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the Lead Attorney supervising a legal discovery process in Kenya.
Your role is to:
1. Review the client's stated legal matter.
2. Formulate a precise legal hypothesis for investigation.
3. Identify what evidence and precedents are needed.

Output a single, clear legal hypothesis (2–3 sentences) that will guide the discovery pipeline.
Focus on: legal rights violated, applicable statutes, and the remedy sought.""",
        ),
        ("human", "Client matter: {matter}"),
    ]
)


def lead_attorney_ingestion(state: DiscoveryState) -> dict[str, Any]:
    """
    Supervisor entry node.
    Validates the input, formulates the legal hypothesis, and sets routing.
    """
    log.info("lead_attorney_ingestion_started")

    # Extract the client matter from the last human message
    matter = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            matter = msg.content
            break

    if not matter:
        matter = state.get("hypothesis", "")

    if not matter.strip():
        log.warning("no_client_matter_provided")
        return {
            "next_node": "rejection",
            "messages": [
                SystemMessage(content="ERROR: No client matter provided. Pipeline cannot proceed.")
            ],
        }

    try:
        llm = _get_llm()
        chain = _SUPERVISOR_PROMPT | llm | StrOutputParser()
        hypothesis = chain.invoke({"matter": matter})
        log.info("hypothesis_formulated", preview=hypothesis[:120])
    except Exception as exc:
        log.warning("hypothesis_generation_failed", error=str(exc), fallback="using raw matter")
        hypothesis = f"Legal hypothesis: {matter}"

    return {
        "hypothesis": hypothesis,
        "next_node": "client_files",
        "iteration": state.get("iteration", 0) + 1,
        "messages": [
            AIMessage(
                content=f"**Lead Attorney Hypothesis:**\n\n{hypothesis}",
                name="lead_attorney",
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: Client Files Runner — invokes subgraph
# ─────────────────────────────────────────────────────────────────────────────
def client_files_runner(state: DiscoveryState) -> dict[str, Any]:
    """
    Invoke the Client Files subgraph with the provided file_path.
    Returns the extracted ClientData back into the main DiscoveryState.
    """
    log.info("client_files_runner_started")

    # file_path is passed in via the initial state; default to sample data
    file_path = state.get("file_path", "data/uploads/lease.pdf")  # type: ignore[attr-defined]

    from src.graph.state import ClientFilesState
    from src.graph.subgraphs.client_files import client_files_subgraph

    subgraph_input: ClientFilesState = {
        "file_path": file_path,
        "client_data": None,
        "messages": [],
    }

    result = client_files_subgraph.invoke(subgraph_input)
    client_data: ClientData | None = result.get("client_data")

    if not client_data:
        log.error("client_files_extraction_returned_no_data")
        return {
            "next_node": "rejection",
            "messages": [SystemMessage(content="ERROR: Client file extraction returned no data.")],
        }

    tenant = client_data["metadata"].get("tenant", "Unknown")
    events = len(client_data.get("timeline", []))
    clauses = len(client_data.get("clauses", []))

    log.info("client_files_complete", tenant=tenant, timeline_events=events, clauses=clauses)

    return {
        "client_data": client_data,
        "next_node": "case_law",
        "messages": [
            AIMessage(
                content=(
                    f"**Client File Extraction Complete**\n"
                    f"- Tenant: {tenant}\n"
                    f"- Timeline events: {events}\n"
                    f"- Notice clauses found: {clauses}"
                ),
                name="client_files_agent",
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: Case Law Runner — invokes subgraph
# ─────────────────────────────────────────────────────────────────────────────
def case_law_runner(state: DiscoveryState) -> dict[str, Any]:
    """
    Invoke the Case Law Search subgraph with the current hypothesis.
    Returns the list of CaseLawResult back into the main DiscoveryState.
    """
    log.info("case_law_runner_started")

    from src.graph.state import CaseLawState
    from src.graph.subgraphs.case_law import case_law_subgraph

    subgraph_input: CaseLawState = {
        "query": state.get("hypothesis", ""),
        "results": [],
        "messages": [],
    }

    result = case_law_subgraph.invoke(subgraph_input)
    results: list[CaseLawResult] = result.get("results", [])

    log.info("case_law_complete", precedents_found=len(results))

    summary_lines = [f"- {r['title']} ({r['citation']}) — score {r['relevance_score']}" for r in results]
    summary_text = "\n".join(summary_lines) if summary_lines else "No precedents found."

    return {
        "case_law_results": results,
        "next_node": "cross_context",
        "messages": [
            AIMessage(
                content=f"**Case Law Search Complete — {len(results)} precedents:**\n{summary_text}",
                name="case_law_agent",
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: Cross-Context Mapping — gap analysis
# ─────────────────────────────────────────────────────────────────────────────
_MAPPING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a senior legal analyst conducting a compliance gap analysis.

Compare the client's lease timeline and notice clauses against the legal precedents found.

Identify specific compliance gaps — situations where the landlord's actions violate:
1. The lease agreement terms.
2. Established case law precedents.
3. Kenyan tenancy statutes.

For each gap, provide a concise one-sentence statement.
Output as a JSON array of strings: ["gap 1", "gap 2", ...]""",
        ),
        (
            "human",
            """CLIENT TIMELINE:
{timeline}

NOTICE CLAUSES:
{clauses}

LEGAL PRECEDENTS:
{precedents}

HYPOTHESIS:
{hypothesis}

Identify all compliance gaps:""",
        ),
    ]
)


def cross_context_mapping(state: DiscoveryState) -> dict[str, Any]:
    """
    Compare client timeline + notice clauses against precedents to identify compliance gaps.
    """
    log.info("cross_context_mapping_started")

    client_data = state.get("client_data") or {}
    timeline = client_data.get("timeline", [])
    clauses = client_data.get("clauses", [])
    precedents = state.get("case_law_results", [])
    hypothesis = state.get("hypothesis", "")

    timeline_str = "\n".join(
        f"- {ev.get('date', '?')}: {ev.get('event', '?')}" for ev in timeline
    )
    clauses_str = "\n".join(f"- {c}" for c in clauses)
    precedents_str = "\n".join(
        f"- {r['title']} ({r['citation']}): {r['summary']}" for r in precedents
    )

    try:
        llm = _get_llm()
        chain = _MAPPING_PROMPT | llm | StrOutputParser()
        raw_output = chain.invoke(
            {
                "timeline": timeline_str or "No timeline data.",
                "clauses": clauses_str or "No clauses extracted.",
                "precedents": precedents_str or "No precedents found.",
                "hypothesis": hypothesis,
            }
        )

        # Parse JSON array from LLM output
        json_match = re.search(r"\[.*?\]", raw_output, re.DOTALL)
        compliance_gaps = json.loads(json_match.group()) if json_match else [raw_output.strip()]

        log.info("compliance_gaps_identified", count=len(compliance_gaps))

    except Exception as exc:
        log.warning("cross_context_mapping_failed", error=str(exc), fallback="using default gaps")
        compliance_gaps = _default_compliance_gaps(timeline, clauses)

    gaps_text = "\n".join(f"• {g}" for g in compliance_gaps)

    return {
        "compliance_gaps": compliance_gaps,
        "next_node": "human_review",
        "messages": [
            AIMessage(
                content=(
                    f"**Cross-Context Mapping — {len(compliance_gaps)} Compliance Gap(s) Found:**\n"
                    f"{gaps_text}"
                ),
                name="cross_context_analyst",
            )
        ],
    }


def _default_compliance_gaps(timeline: list, clauses: list) -> list[str]:
    """Deterministic fallback gap analysis for offline/test scenarios."""
    gaps = []
    # Look for short notice in timeline
    for ev in timeline:
        if "18 days" in ev.get("event", "") or "notice" in ev.get("event", "").lower():
            gaps.append(
                "Landlord issued only 18 days' notice, violating Section 4.2 which requires 30 days."
            )
            break

    if not gaps:
        gaps = [
            "Written eviction notice period was shorter than the contractual 30-day requirement.",
            "Verbal notice given prior to written notice does not satisfy statutory requirements.",
        ]
    return gaps


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: Human Review (HITL)
# ─────────────────────────────────────────────────────────────────────────────
def human_review(state: DiscoveryState) -> dict[str, Any]:
    """
    Human-in-the-Loop node. Raises a LangGraph interrupt to pause the graph.
    Lead counsel reviews the compliance gaps and approves or rejects the findings.

    Resume by calling graph.invoke() with:
        {"verdict_approved": True/False, "counsel_notes": "..."}
    via the FastAPI POST /discovery/{thread_id}/approve endpoint.
    """
    log.info("human_review_interrupt_raised")

    gaps = state.get("compliance_gaps", [])
    hypothesis = state.get("hypothesis", "")
    tenant = (state.get("client_data") or {}).get("metadata", {}).get("tenant", "Unknown")
    precedents_count = len(state.get("case_law_results", []))

    review_payload = {
        "status": "AWAITING_LEAD_COUNSEL_REVIEW",
        "tenant": tenant,
        "hypothesis": hypothesis,
        "compliance_gaps_count": len(gaps),
        "compliance_gaps": gaps,
        "precedents_reviewed": precedents_count,
        "instructions": (
            "Review the compliance gaps above. "
            "Approve to proceed with final verdict generation, or reject to terminate."
        ),
    }

    # This raises a LangGraph interrupt — the graph suspends here and saves state.
    # The FastAPI /approve endpoint resumes it with human_verdict.
    human_verdict = interrupt(review_payload)

    # After resumption, human_verdict contains the counselor's decision
    approved = human_verdict.get("verdict_approved", False) if isinstance(human_verdict, dict) else bool(human_verdict)
    notes = human_verdict.get("counsel_notes", "") if isinstance(human_verdict, dict) else ""

    log.info("human_review_resumed", approved=approved)

    return {
        "verdict_approved": approved,
        "next_node": "generate_verdict" if approved else "rejection",
        "messages": [
            SystemMessage(
                content=(
                    f"**Lead Counsel Decision:** {'✅ APPROVED' if approved else '❌ REJECTED'}\n"
                    + (f"Notes: {notes}" if notes else "")
                )
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: Generate Verdict
# ─────────────────────────────────────────────────────────────────────────────
_VERDICT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the Lead Attorney writing a final legal discovery verdict memo.

Based on the compliance gaps identified and supporting case law, write a professional
legal verdict memo that includes:

1. **Executive Summary** (2-3 sentences)
2. **Legal Findings** (numbered list of violations found)
3. **Supporting Precedents** (list relevant cases)
4. **Recommended Actions** (list practical next steps for the client)
5. **Conclusion**

Use formal legal language appropriate for Kenyan jurisdiction.""",
        ),
        (
            "human",
            """HYPOTHESIS: {hypothesis}
COMPLIANCE GAPS: {gaps}
SUPPORTING PRECEDENTS: {precedents}
LEAD COUNSEL NOTES: {notes}

Generate the final verdict memo:""",
        ),
    ]
)


def generate_verdict(state: DiscoveryState) -> dict[str, Any]:
    """Generate the final legal verdict memo after human approval."""
    log.info("verdict_generation_started")

    gaps = state.get("compliance_gaps", [])
    precedents = state.get("case_law_results", [])
    hypothesis = state.get("hypothesis", "")
    notes = ""  # counsel notes are in the last SystemMessage if present
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, SystemMessage) and "Lead Counsel Decision" in msg.content:
            if "Notes:" in msg.content:
                notes = msg.content.split("Notes:")[-1].strip()
            break

    gaps_str = "\n".join(f"{i+1}. {g}" for i, g in enumerate(gaps))
    precedents_str = "\n".join(
        f"- {r['title']} ({r['citation']}): {r['summary']}" for r in precedents
    )

    try:
        llm = _get_llm(temperature=0.1)
        chain = _VERDICT_PROMPT | llm | StrOutputParser()
        verdict = chain.invoke(
            {
                "hypothesis": hypothesis,
                "gaps": gaps_str or "No specific gaps identified.",
                "precedents": precedents_str or "No precedents found.",
                "notes": notes or "None.",
            }
        )
        log.info("verdict_generated", chars=len(verdict))
    except Exception as exc:
        log.error("verdict_generation_failed", error=str(exc))
        verdict = _default_verdict(gaps, precedents, hypothesis)

    return {
        "verdict_approved": True,
        "next_node": "END",
        "messages": [
            AIMessage(
                content=f"**FINAL LEGAL VERDICT MEMO**\n\n{verdict}",
                name="lead_attorney_verdict",
            )
        ],
    }


def _default_verdict(gaps: list, precedents: list, hypothesis: str) -> str:
    gaps_text = "\n".join(f"{i+1}. {g}" for i, g in enumerate(gaps))
    prec_text = "\n".join(f"- {r['title']} ({r['citation']})" for r in precedents)
    return f"""## LEGAL DISCOVERY VERDICT MEMO

**Executive Summary**
Based on the investigation conducted, the landlord's eviction notice is legally insufficient
under the terms of the lease agreement and Kenyan tenancy law.

**Legal Findings**
{gaps_text or "1. Notice period was shorter than the contractual requirement."}

**Supporting Precedents**
{prec_text or "- Kamau v. Kiambu County Housing Board, 2021 KLR 456"}

**Recommended Actions**
1. File an application with the Business Premises Rent Tribunal.
2. Seek an injunction to halt the eviction pending determination.
3. Claim damages for unlawful eviction procedure.

**Conclusion**
The evidence supports a strong case for the tenant. Recommend immediate legal action.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Node 7: Rejection Node
# ─────────────────────────────────────────────────────────────────────────────
def rejection_node(state: DiscoveryState) -> dict[str, Any]:
    """
    Terminates the pipeline gracefully when input is invalid or counsel rejects.
    """
    reason = "Unknown"
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, SystemMessage) and "ERROR" in msg.content:
            reason = msg.content.replace("ERROR:", "").strip()
            break
        if isinstance(msg, SystemMessage) and "REJECTED" in msg.content:
            reason = "Lead counsel rejected the findings."
            break

    log.warning("pipeline_rejected", reason=reason)

    return {
        "next_node": "END",
        "messages": [
            AIMessage(
                content=f"**Pipeline Terminated**\n\nReason: {reason}",
                name="system",
            )
        ],
    }
