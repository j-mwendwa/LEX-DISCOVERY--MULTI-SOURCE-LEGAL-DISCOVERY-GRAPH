"""
src/vectordb/qdrant_store.py — Qdrant vector store with hybrid search support.

Provides:
  - QdrantVectorStore: wraps QdrantClient with collection management
  - ensure_collection(): idempotent collection creation with dense+sparse vectors
  - hybrid_search(): combines dense ANN + sparse BM25 with reciprocal rank fusion
"""
from __future__ import annotations

from typing import Any

from src.core.exceptions import ConfigError, SearchError
from src.core.logging import get_logger

log = get_logger(__name__)


class QdrantVectorStore:
    """
    Thin wrapper around QdrantClient for LEX-DISCOVERY.

    Supports:
      - Dense vector storage (BAAI/bge-small-en-v1.5, dim=384)
      - Hybrid search (dense + sparse BM25) via Qdrant native hybrid mode
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str = "",
        collection_name: str = "case_law_precedents",
        vector_size: int = 384,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = None

    # ──────────────────────────────────────────────────────────────────────────
    # Client lifecycle
    # ──────────────────────────────────────────────────────────────────────────
    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient

                kwargs: dict[str, Any] = {"url": self.url}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                self._client = QdrantClient(**kwargs)
                log.info("qdrant_client_connected", url=self.url)
            except ImportError:
                raise ConfigError(
                    "qdrant-client is not installed. Run: pip install qdrant-client"
                ) from None
            except Exception as exc:
                raise SearchError(f"Failed to connect to Qdrant at {self.url}: {exc}") from exc
        return self._client

    # ──────────────────────────────────────────────────────────────────────────
    # Collection management
    # ──────────────────────────────────────────────────────────────────────────
    def ensure_collection(self) -> None:
        """
        Idempotent: create the collection if it doesn't exist.
        Configures both dense (Cosine) and sparse (BM25) vectors for hybrid search.
        """
        from qdrant_client.models import (
            Distance,
            SparseVectorParams,
            VectorParams,
        )

        client = self._get_client()
        existing = {c.name for c in client.get_collections().collections}
        if self.collection_name in existing:
            log.debug("qdrant_collection_exists", collection=self.collection_name)
            return

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
            sparse_vectors_config={
                "bm25": SparseVectorParams(),
            },
        )
        log.info(
            "qdrant_collection_created",
            collection=self.collection_name,
            vector_size=self.vector_size,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Hybrid Search
    # ──────────────────────────────────────────────────────────────────────────
    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid (dense + sparse BM25) search and return ranked results.

        Args:
            query_vector: Dense embedding of the query.
            query_text: Raw text for sparse BM25 tokenisation.
            top_k: Maximum results to return.
            score_threshold: Minimum score to include a result.

        Returns:
            List of dicts with keys: id, score, payload.
        """
        client = self._get_client()
        try:
            from qdrant_client.models import (
                FusionQuery,
                NamedSparseVector,
                Prefetch,
                SparseVector,
            )

            # Dense prefetch
            dense_prefetch = Prefetch(
                query=query_vector,
                using="",
                limit=top_k * 2,
            )

            # Sparse BM25 prefetch (tokenised query)
            sparse_indices, sparse_values = _bm25_encode(query_text)
            sparse_prefetch = Prefetch(
                query=NamedSparseVector(
                    name="bm25",
                    vector=SparseVector(indices=sparse_indices, values=sparse_values),
                ),
                limit=top_k * 2,
            )

            results = client.query_points(
                collection_name=self.collection_name,
                prefetch=[dense_prefetch, sparse_prefetch],
                query=FusionQuery(fusion="rrf"),  # Reciprocal Rank Fusion
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

            formatted = []
            for point in results.points:
                formatted.append(
                    {
                        "id": str(point.id),
                        "score": point.score,
                        "payload": point.payload or {},
                    }
                )

            log.info(
                "qdrant_hybrid_search_complete",
                query_preview=query_text[:80],
                results_count=len(formatted),
            )
            return formatted

        except Exception as exc:
            log.error("qdrant_search_error", error=str(exc))
            raise SearchError(f"Qdrant hybrid search failed: {exc}") from exc

    # ──────────────────────────────────────────────────────────────────────────
    # Upsert
    # ──────────────────────────────────────────────────────────────────────────
    def upsert(self, points: list[dict[str, Any]]) -> None:
        """
        Upsert a list of points into the collection.

        Each point must have: id (str/int), vector (List[float]), payload (dict).
        Optionally: sparse_vector {"indices": List[int], "values": List[float]}.
        """
        from qdrant_client.models import PointStruct, SparseVector

        client = self._get_client()
        structs = []
        for p in points:
            vectors: Any = {"": p["vector"]}
            if sv := p.get("sparse_vector"):
                vectors["bm25"] = SparseVector(
                    indices=sv["indices"], values=sv["values"]
                )
            structs.append(
                PointStruct(id=p["id"], vector=vectors, payload=p.get("payload", {}))
            )

        client.upsert(collection_name=self.collection_name, points=structs)
        log.info("qdrant_upsert_complete", count=len(structs))


# ─────────────────────────────────────────────────────────────────────────────
# Minimal BM25 encoder (no external dependency)
# ─────────────────────────────────────────────────────────────────────────────
def _bm25_encode(text: str) -> tuple[list[int], list[float]]:
    """
    Produce a sparse BM25 vector from raw text using simple term-frequency scoring.
    For production, replace with a proper BM25 tokeniser (e.g. rank_bm25).
    """
    import math
    import re
    from collections import Counter

    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return [], []

    tf = Counter(tokens)
    total = len(tokens)
    # Unique sorted token indices (hash-based, deterministic)
    vocab: dict[str, int] = {}
    for tok in sorted(tf.keys()):
        vocab[tok] = abs(hash(tok)) % (2**24)

    indices = sorted(vocab[tok] for tok in tf)
    values = [
        (1 + math.log(tf[tok] / total + 1)) for tok in sorted(tf.keys())
    ]
    return indices, values


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────
def get_qdrant_store(
    url: str = "http://localhost:6333",
    api_key: str = "",
    collection_name: str = "case_law_precedents",
) -> QdrantVectorStore:
    """Instantiate and return a QdrantVectorStore."""
    return QdrantVectorStore(url=url, api_key=api_key, collection_name=collection_name)
