"""
src/ingestion/llamaindex_pipeline.py — LlamaIndex-powered PDF ingestion for LEX-DISCOVERY.

Flow:
  ingest_lease_pdf(file_path, collection) →
    1. Validate path (security: no traversal outside allowed roots)
    2. SimpleDirectoryReader → load documents
    3. SentenceSplitter(chunk_size=512, overlap=64)
    4. Embed with BAAI/bge-small-en-v1.5
    5. Upsert into Qdrant collection via QdrantVectorStore
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List, Optional

from src.core.exceptions import IngestionError
from src.core.logging import get_logger

log = get_logger(__name__)

_ALLOWED_INGEST_ROOTS = (Path("data").resolve(),)


def _validate_path(file_path: str) -> Path:
    """Ensure the file is within an allowed root (prevents path traversal)."""
    resolved = Path(file_path).resolve()
    for root in _ALLOWED_INGEST_ROOTS:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise IngestionError(
        f"Path '{file_path}' is outside allowed ingestion roots: "
        f"{[str(r) for r in _ALLOWED_INGEST_ROOTS]}"
    )


def ingest_lease_pdf(
    file_path: str,
    collection_name: str = "case_law_precedents",
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: str = "",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    skip_path_validation: bool = False,
) -> int:
    """
    Ingest a lease PDF into Qdrant.

    Args:
        file_path: Path to the PDF file or directory.
        collection_name: Target Qdrant collection.
        qdrant_url: Qdrant server URL.
        qdrant_api_key: Optional Qdrant API key.
        chunk_size: Tokens per chunk.
        chunk_overlap: Overlap between chunks.
        skip_path_validation: Set True in tests to bypass root check.

    Returns:
        Number of chunks upserted.
    """
    if not skip_path_validation:
        validated = _validate_path(file_path)
    else:
        validated = Path(file_path)

    if not validated.exists():
        raise IngestionError(f"Path does not exist: {validated}")

    try:
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import SentenceSplitter
    except ImportError as exc:
        raise IngestionError(
            f"LlamaIndex is not installed: {exc}. "
            "Run: pip install llama-index-core llama-index-readers-file"
        ) from exc

    log.info("ingestion_started", path=str(validated))

    # Load documents
    input_dir = str(validated) if validated.is_dir() else str(validated.parent)
    input_files = None if validated.is_dir() else [str(validated)]

    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        input_files=input_files,
        required_exts=[".pdf", ".txt"],
        recursive=True,
    )
    documents = reader.load_data()
    if not documents:
        raise IngestionError(f"No documents found at: {validated}")

    log.info("ingestion_docs_loaded", count=len(documents))

    # Chunk
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        ]
    )
    nodes = pipeline.run(documents=documents)
    log.info("ingestion_chunks_created", count=len(nodes))

    # Embed and upsert
    from llama_index.core import Settings as LISettings

    embed_model = LISettings.embed_model
    if embed_model is None:
        raise IngestionError(
            "LlamaIndex embed_model is not configured. Call setup_llamaindex() first."
        )

    from src.vectordb.qdrant_store import get_qdrant_store

    store = get_qdrant_store(
        url=qdrant_url, api_key=qdrant_api_key, collection_name=collection_name
    )
    store.ensure_collection()

    points = []
    for node in nodes:
        embedding = embed_model.get_text_embedding(node.get_content())
        points.append(
            {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "text": node.get_content(),
                    "doc_id": node.node_id,
                    "source": str(validated),
                    "metadata": node.metadata,
                },
            }
        )

    store.upsert(points)
    log.info("ingestion_complete", chunks_upserted=len(points))
    return len(points)
