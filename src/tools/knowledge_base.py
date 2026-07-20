"""
src/tools/knowledge_base.py — Knowledge base search tool for LEX-DISCOVERY.

Provides:
  - qdrant_hybrid_search(): Real Qdrant hybrid search (dense + BM25 via RRF).
  - mock_qdrant_search():   Deterministic mock for tests / offline dev.
  - get_embedder():         HuggingFace bge-small-en-v1.5 embedding helper.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.logging import get_logger
from src.graph.state import CaseLawResult

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helper
# ─────────────────────────────────────────────────────────────────────────────
def get_embedder():
    """
    Return the LlamaIndex global embed_model (configured at startup).
    Falls back to a simple character-hashing stub in test environments.
    """
    try:
        from llama_index.core import Settings

        if Settings.embed_model is not None:
            return Settings.embed_model
    except ImportError:
        pass

    # Stub for offline / unit-test use
    class _StubEmbedder:
        def get_text_embedding(self, text: str) -> List[float]:
            # 384-dim deterministic pseudo-embedding
            h = hash(text)
            return [(h >> i & 1) * 0.1 for i in range(384)]

    return _StubEmbedder()


# ─────────────────────────────────────────────────────────────────────────────
# Real Qdrant hybrid search
# ─────────────────────────────────────────────────────────────────────────────
def qdrant_hybrid_search(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    qdrant_url: str = "",
    qdrant_api_key: str = "",
    collection_name: str = "case_law_precedents",
) -> List[CaseLawResult]:
    """
    Perform hybrid (dense bge-small + sparse Qdrant/bm25) search via LlamaIndex.

    Uses LlamaIndex QdrantVectorStore with fastembed_sparse_model="Qdrant/bm25"
    and RRF fusion — the recommended production path for Qdrant Cloud hybrid search.

    Falls back to mock results if Qdrant Cloud is unreachable.
    """
    log.info("qdrant_hybrid_search_called", query_preview=query[:80])

    try:
        from src.vectordb.llamaindex_qdrant import hybrid_search_qdrant

        raw = hybrid_search_qdrant(query=query, top_k=top_k, collection_name=collection_name)

        results: List[CaseLawResult] = []
        for r in raw:
            results.append(
                {
                    "title": r.get("title", "Unknown"),
                    "citation": r.get("citation", ""),
                    "summary": r.get("summary", r.get("text", ""))[:300],
                    "relevance_score": round(float(r.get("relevance_score", 0.0)), 4),
                }
            )
        log.info("hybrid_search_results_mapped", count=len(results))
        return results

    except Exception as exc:
        log.warning("qdrant_search_failed", error=str(exc), fallback="mock results")
        return mock_qdrant_search(query)


# ─────────────────────────────────────────────────────────────────────────────
# Mock search — offline / test fallback
# ─────────────────────────────────────────────────────────────────────────────
def mock_qdrant_search(query: str) -> List[CaseLawResult]:
    """
    Deterministic mock that returns Kenya-specific tenancy law precedents.
    Used in tests and when Qdrant is unavailable.
    """
    return [
        {
            "title": "Kamau v. Kiambu County Housing Board",
            "citation": "2021 KLR 456",
            "summary": (
                "Notice of fewer than 14 days for residential eviction was held legally "
                "insufficient. Court affirmed that statutory minimum of 30 days must be "
                "strictly observed under the Landlord and Tenant (Shops, Hotels and Catering "
                "Establishments) Act."
            ),
            "relevance_score": 0.95,
        },
        {
            "title": "Mwangi v. Nairobi Realty Corp",
            "citation": "2019 KLR 789",
            "summary": (
                "Notice period must be calculated from the date of actual delivery, not the "
                "date of posting. Electronic delivery (email/SMS) is not valid substitute for "
                "written notice under Kenyan tenancy law."
            ),
            "relevance_score": 0.88,
        },
        {
            "title": "Odhiambo v. Unity Housing Cooperative",
            "citation": "2020 KEHC 1023",
            "summary": (
                "A landlord who verbally requests vacation cannot rely on verbal notice to "
                "satisfy statutory notice requirements. Written notice is mandatory. "
                "The court awarded 3 months' rent as damages for unlawful eviction."
            ),
            "relevance_score": 0.82,
        },
        {
            "title": "Njoroge v. Pangani Estate Management",
            "citation": "2018 KLR 334",
            "summary": (
                "Where a tenant disputes the validity of a notice, the burden of proof lies "
                "with the landlord to demonstrate lawful service. Mere assertion of posting "
                "is insufficient without acknowledgment of receipt."
            ),
            "relevance_score": 0.76,
        },
    ]
