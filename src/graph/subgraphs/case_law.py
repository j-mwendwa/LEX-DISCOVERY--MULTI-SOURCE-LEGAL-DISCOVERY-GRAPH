"""
src/graph/subgraphs/case_law.py — Case Law Search Subgraph (Phase 3).

Nodes:
  1. query_generation_node  — LLM refines the raw hypothesis into targeted legal queries.
  2. search_precedents_node — Qdrant Hybrid Search (dense + BM25 + RRF) for precedents.
  3. rerank_node            — Re-scores and deduplicates results by relevance.

The subgraph operates on CaseLawState (isolated from the main graph).
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.config import cfg, settings
from src.core.exceptions import SearchError
from src.core.logging import get_logger
from src.graph.state import CaseLawResult, CaseLawState
from src.tools.knowledge_base import qdrant_hybrid_search

log = get_logger(__name__)

_QUERY_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Kenyan legal research specialist. Given a legal hypothesis about a tenancy dispute,
generate 2–3 precise search queries optimised for finding relevant case law precedents.

Focus on:
- Specific legal issues (notice period, eviction validity, written vs verbal notice)
- Kenyan tenancy statutes (Landlord and Tenant Act, Rent Restriction Act)
- Court-held principles from Kenyan Law Reports (KLR)

Output ONLY the queries, one per line, no numbering or preamble.""",
        ),
        ("human", "Legal hypothesis: {hypothesis}"),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: LLM Query Generation
# ─────────────────────────────────────────────────────────────────────────────
def query_generation_node(state: CaseLawState) -> Dict[str, Any]:
    """
    Refines the raw search query / legal hypothesis into targeted precedent search queries.
    """
    log.info("query_generation_started", query=state["query"])

    model_name = cfg.get("llm", {}).get("default_model", "gemini-2.0-flash")

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            google_api_key=settings.google_api_key or None,
        )
        chain = _QUERY_GEN_PROMPT | llm | StrOutputParser()
        generated = chain.invoke({"hypothesis": state["query"]})
        queries = [q.strip() for q in generated.strip().splitlines() if q.strip()]
        log.info("queries_generated", count=len(queries), queries=queries)
    except Exception as exc:
        log.warning("query_generation_failed", error=str(exc), fallback="using original query")
        queries = [state["query"]]

    # Store generated queries back as the refined query (joined)
    refined_query = "; ".join(queries) if queries else state["query"]

    return {
        "query": refined_query,
        "messages": [
            SystemMessage(
                content=f"Generated {len(queries)} search queries: {'; '.join(queries)}"
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: Qdrant Hybrid Search
# ─────────────────────────────────────────────────────────────────────────────
def search_precedents_node(state: CaseLawState) -> Dict[str, Any]:
    """
    Execute Qdrant hybrid search for each sub-query and aggregate results.
    """
    log.info("precedent_search_started")

    top_k = cfg.get("retrieval", {}).get("top_k", 5)
    score_threshold = cfg.get("retrieval", {}).get("similarity_cutoff", 0.0)

    qdrant_url = cfg.get("qdrant", {}).get("url", "http://localhost:6333")
    collection = cfg.get("qdrant", {}).get("collection_name", "case_law_precedents")

    # Split multi-query back into individual queries
    sub_queries = [q.strip() for q in state["query"].split(";") if q.strip()]

    all_results: List[CaseLawResult] = []
    seen_citations: set = set()

    for sq in sub_queries:
        try:
            results = qdrant_hybrid_search(
                query=sq,
                top_k=top_k,
                score_threshold=score_threshold,
                qdrant_url=qdrant_url,
                qdrant_api_key=settings.qdrant_api_key,
                collection_name=collection,
            )
            for r in results:
                if r["citation"] not in seen_citations:
                    seen_citations.add(r["citation"])
                    all_results.append(r)
        except SearchError as exc:
            log.error("search_error_for_query", query=sq, error=str(exc))

    log.info("precedent_search_complete", total_unique_results=len(all_results))

    return {
        "results": all_results,
        "messages": [
            SystemMessage(
                content=(
                    f"Qdrant hybrid search complete. "
                    f"Found {len(all_results)} unique precedents across {len(sub_queries)} queries."
                )
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: Re-ranking
# ─────────────────────────────────────────────────────────────────────────────
def rerank_node(state: CaseLawState) -> Dict[str, Any]:
    """
    Re-rank results by relevance_score (descending) and cap at top_k.
    In production, replace with a cross-encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2).
    """
    top_k = cfg.get("retrieval", {}).get("top_k", 5)
    results = state.get("results", [])

    reranked = sorted(results, key=lambda r: r["relevance_score"], reverse=True)[:top_k]

    log.info("reranking_complete", kept=len(reranked), original=len(results))

    return {
        "results": reranked,
        "messages": [
            SystemMessage(
                content=(
                    f"Re-ranked and deduplicated to top {len(reranked)} precedents. "
                    f"Top result: {reranked[0]['title'] if reranked else 'None'} "
                    f"(score: {reranked[0]['relevance_score'] if reranked else 'N/A'})"
                )
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph Construction
# ─────────────────────────────────────────────────────────────────────────────
def build_case_law_graph():
    """Build and compile the isolated Case Law search subgraph."""
    workflow = StateGraph(CaseLawState)

    workflow.add_node("generate_query", query_generation_node)
    workflow.add_node("search", search_precedents_node)
    workflow.add_node("rerank", rerank_node)

    workflow.set_entry_point("generate_query")
    workflow.add_edge("generate_query", "search")
    workflow.add_edge("search", "rerank")
    workflow.add_edge("rerank", END)

    return workflow.compile()


# Module-level singleton — imported by the main graph
case_law_subgraph = build_case_law_graph()
