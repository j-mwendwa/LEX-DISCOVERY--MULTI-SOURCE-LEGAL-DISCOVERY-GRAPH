"""
src/vectordb/llamaindex_qdrant.py — LlamaIndex + Qdrant hybrid store.

Architecture: QdrantVectorStoreWrapper class that natively leverages LlamaIndex's
QdrantVectorStore with:
  - Dense vectors:  BAAI/bge-small-en-v1.5 via fastembed (dim=384)
  - Sparse vectors: Qdrant/bm25 via fastembed (no external BM25 call needed)
  - Hybrid search:  Reciprocal Rank Fusion (RRF) — native Qdrant feature

This module is the single point of truth for all Qdrant Cloud interactions
via LlamaIndex. Use it for both ingestion and retrieval.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

import qdrant_client
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.config import cfg, settings
from src.core.logging import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# QdrantVectorStoreWrapper — primary class (new architecture)
# ─────────────────────────────────────────────────────────────────────────────
class QdrantVectorStoreWrapper:
    """
    High-performance wrapper around LlamaIndex's QdrantVectorStore.

    Natively supports dense embeddings and BM25 hybrid search via FastEmbed.
    Delegates all low-level Qdrant operations to LlamaIndex's abstraction layer.

    Attributes:
        store: The underlying LlamaIndex QdrantVectorStore instance.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str = "",
        collection_name: str = "case_law_precedents",
    ) -> None:
        # Initialize standard Qdrant Client lifecycle cleanly
        self._client = qdrant_client.QdrantClient(
            url=url,
            api_key=api_key or None,
            prefer_grpc=True,   # gRPC for faster Cloud throughput
            timeout=30,
        )

        # Instantiate LlamaIndex vector store with native hybrid configuration flags
        self.store = QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",   # Production-grade BM25 local encoder
        )
        log.info("qdrant_indexed_store_initialized", collection=collection_name)

    # ──────────────────────────────────────────────────────────────────────────
    # Hybrid Search
    # ──────────────────────────────────────────────────────────────────────────
    def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Execute an advanced hybrid search combining dense and sparse criteria.

        Uses LlamaIndex's VectorStoreQuery with HYBRID mode, which triggers
        RRF fusion of dense (bge-small) and sparse (BM25) rankings natively.

        Args:
            query_vector: Dense embedding of the query (dim must match collection).
            query_text:   Raw query text for BM25 sparse encoding via fastembed.
            top_k:        Maximum number of results to return.

        Returns:
            List of dicts with keys: id, score, payload.
        """
        # Formulate a structured LlamaIndex vector store query object
        query = VectorStoreQuery(
            query_embedding=query_vector,
            query_str=query_text,
            mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=top_k,
        )

        # Execute query against LlamaIndex vector store abstraction
        query_result = self.store.query(query)

        formatted_results: List[Dict[str, Any]] = []
        if query_result.nodes:
            for node, score in zip(query_result.nodes, query_result.scores or []):
                formatted_results.append(
                    {
                        "id": node.node_id,
                        "score": score,
                        "payload": node.metadata or {},
                    }
                )

        log.info("qdrant_hybrid_search_complete", results_count=len(formatted_results))
        return formatted_results

    # ──────────────────────────────────────────────────────────────────────────
    # Upsert
    # ──────────────────────────────────────────────────────────────────────────
    def upsert(self, points: List[Dict[str, Any]]) -> None:
        """
        Upsert a list of structured document points into the collection.

        Each point must have:
            - id (str | int): Unique identifier.
            - vector (List[float]): Dense embedding.
            - payload (dict): Metadata fields (must include ``text`` for BM25).

        Args:
            points: List of point dicts conforming to the schema above.
        """
        nodes: List[TextNode] = []
        for p in points:
            # Wrap standard points into LlamaIndex TextNodes
            node = TextNode(
                id_=str(p["id"]),
                text=p.get("payload", {}).get("text", ""),
                embedding=p["vector"],
                metadata=p.get("payload", {}),
            )
            nodes.append(node)

        self.store.add(nodes)
        log.info("qdrant_upsert_complete", count=len(nodes))


# ─────────────────────────────────────────────────────────────────────────────
# Singleton Qdrant client (used by module-level helpers)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_qdrant_client() -> qdrant_client.QdrantClient:
    """Return a singleton QdrantClient connected to Qdrant Cloud."""
    try:
        client = qdrant_client.QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            prefer_grpc=True,
            timeout=30,
        )
        # Connectivity check
        client.get_collections()
        log.info("qdrant_cloud_connected", url=settings.qdrant_url)
        return client
    except Exception as exc:
        log.error(
            "qdrant_cloud_connection_failed",
            url=settings.qdrant_url,
            error=str(exc),
        )
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Module-level factory (returns raw LlamaIndex store backed by project config)
# ─────────────────────────────────────────────────────────────────────────────
def get_llamaindex_vector_store(
    collection_name: Optional[str] = None,
    enable_hybrid: bool = True,
) -> QdrantVectorStore:
    """
    Build a raw LlamaIndex QdrantVectorStore backed by project config.

    Prefer ``get_wrapper()`` for typical usage; use this only when you need
    direct access to the LlamaIndex store without the wrapper.

    Args:
        collection_name: Qdrant collection. Defaults to config value.
        enable_hybrid:   Enable hybrid dense+sparse search.

    Returns:
        LlamaIndex QdrantVectorStore instance.
    """
    col = collection_name or cfg.get("qdrant", {}).get(
        "collection_name", "case_law_precedents"
    )
    sparse_model = cfg.get("qdrant", {}).get("sparse_model", "Qdrant/bm25")
    client = _get_qdrant_client()

    store = QdrantVectorStore(
        client=client,
        collection_name=col,
        enable_hybrid=enable_hybrid,
        fastembed_sparse_model=sparse_model,
        alpha=0.3,
    )

    log.info(
        "llamaindex_qdrant_store_created",
        collection=col,
        hybrid=enable_hybrid,
        sparse_model=sparse_model,
    )
    return store


def get_wrapper(
    collection_name: Optional[str] = None,
) -> QdrantVectorStoreWrapper:
    """
    Instantiate a ``QdrantVectorStoreWrapper`` backed by project config.

    Args:
        collection_name: Qdrant collection. Defaults to config value.

    Returns:
        Configured QdrantVectorStoreWrapper ready for upsert and hybrid search.
    """
    col = collection_name or cfg.get("qdrant", {}).get(
        "collection_name", "case_law_precedents"
    )
    return QdrantVectorStoreWrapper(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or "",
        collection_name=col,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VectorStoreIndex (for retrieval via LlamaIndex query engine)
# ─────────────────────────────────────────────────────────────────────────────
def get_vector_index(collection_name: Optional[str] = None):
    """
    Load an existing LlamaIndex VectorStoreIndex from Qdrant.

    Returns:
        VectorStoreIndex ready for hybrid retrieval.
    """
    try:
        from llama_index.core import VectorStoreIndex
        from llama_index.core.storage.storage_context import StorageContext
    except ImportError:
        raise ImportError("Install: pip install llama-index-core")

    store = get_llamaindex_vector_store(
        collection_name=collection_name, enable_hybrid=True
    )
    storage_context = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=store,
        storage_context=storage_context,
    )
    log.info("vector_index_loaded", collection=collection_name)
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Search (module-level helper — bypasses LlamaIndex query engine)
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_search_qdrant(
    query: str,
    top_k: int = 5,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform LlamaIndex-powered hybrid search against Qdrant Cloud.

    Combines dense (bge-small-en-v1.5) + sparse (Qdrant/bm25) with RRF.

    Args:
        query:           Natural language query.
        top_k:           Maximum results.
        collection_name: Target collection.

    Returns:
        List of dicts: {title, citation, summary, relevance_score, text}
    """
    try:
        index = get_vector_index(collection_name=collection_name)

        retriever = index.as_retriever(
            similarity_top_k=top_k,
            sparse_top_k=top_k * 2,          # Sparse prefetch buffer
            vector_store_query_mode="hybrid",  # RRF fusion
        )

        nodes = retriever.retrieve(query)
        log.info(
            "hybrid_search_complete",
            query_preview=query[:80],
            results=len(nodes),
        )

        results: List[Dict[str, Any]] = []
        for node in nodes:
            meta = node.metadata or {}
            results.append(
                {
                    "title": meta.get("title", meta.get("source", "Unknown")),
                    "citation": meta.get("citation", ""),
                    "summary": node.get_content()[:300],
                    "relevance_score": round(float(node.score or 0.0), 4),
                    "text": node.get_content(),
                }
            )
        return results

    except Exception as exc:
        log.warning("hybrid_search_failed", error=str(exc), fallback="mock results")
        from src.tools.knowledge_base import mock_qdrant_search

        return mock_qdrant_search(query)  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Ensure collection exists with hybrid config
# ─────────────────────────────────────────────────────────────────────────────
def ensure_hybrid_collection(collection_name: Optional[str] = None) -> None:
    """
    Create the Qdrant collection with dense + sparse vector config if it doesn't exist.
    Called automatically during ingestion.
    """
    from qdrant_client.models import Distance, SparseVectorParams, VectorParams

    col = collection_name or cfg.get("qdrant", {}).get(
        "collection_name", "case_law_precedents"
    )
    vector_size = cfg.get("qdrant", {}).get("vector_size", 384)
    client = _get_qdrant_client()

    existing = {c.name for c in client.get_collections().collections}
    if col in existing:
        log.debug("collection_already_exists", collection=col)
        return

    client.create_collection(
        collection_name=col,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        sparse_vectors_config={"bm25": SparseVectorParams()},
    )
    log.info("hybrid_collection_created", collection=col, vector_size=vector_size)
