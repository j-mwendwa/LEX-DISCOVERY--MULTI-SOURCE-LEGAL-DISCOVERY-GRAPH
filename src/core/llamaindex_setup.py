"""
src/core/llamaindex_setup.py — LlamaIndex global settings for LEX-DISCOVERY.

Configures:
  - Dense embeddings: BAAI/bge-small-en-v1.5 (dim=384, local via fastembed)
  - Sparse embeddings: Qdrant/bm25 (via fastembed — no extra calls)
  - LLM: HuggingFace Saul-7B-Instruct-v1 → Gemini fallback

The fastembed backend is used by LlamaIndex's QdrantVectorStore for hybrid
search without requiring a separate Qdrant sparse model API call.
"""
from __future__ import annotations

from src.core.logging import get_logger

log = get_logger(__name__)


def setup_llamaindex(
    google_api_key: str = "",
    hf_api_key: str = "",
    hf_model_id: str = "Equall/Saul-7B-Instruct-v1",
) -> None:
    """
    Configure LlamaIndex global settings.
    Must be called once at application startup before any ingestion or retrieval.
    """
    try:
        from llama_index.core import Settings

        # ── Dense embeddings via fastembed (local, no GPU needed) ─────────────
        try:
            from llama_index.embeddings.fastembed import FastEmbedEmbedding

            Settings.embed_model = FastEmbedEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
            )
            log.info(
                "llamaindex_dense_embed_set",
                model="BAAI/bge-small-en-v1.5",
                backend="fastembed",
                dim=384,
            )
        except ImportError:
            # Fallback: HuggingFace embedding (requires transformers + torch)
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            log.info(
                "llamaindex_dense_embed_set",
                model="BAAI/bge-small-en-v1.5",
                backend="huggingface",
                dim=384,
            )

        # ── LLM: HuggingFace Saul-7B → Gemini fallback ───────────────────────
        _configure_llm(Settings, hf_api_key, hf_model_id, google_api_key)

        log.info("llamaindex_setup_complete")

    except ImportError as exc:
        log.warning(
            "llamaindex_setup_skipped",
            reason=str(exc),
            hint=(
                "Install: pip install llama-index-core "
                "llama-index-embeddings-fastembed "
                "llama-index-vector-stores-qdrant"
            ),
        )


def _configure_llm(Settings, hf_api_key: str, hf_model_id: str, google_api_key: str) -> None:
    """Configure LlamaIndex LLM with HuggingFace Saul → Gemini fallback."""

    # Try HuggingFace Saul Legal AI
    if hf_api_key:
        try:
            from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

            Settings.llm = HuggingFaceInferenceAPI(
                model_name=hf_model_id,
                token=hf_api_key,
                context_window=4096,
                num_output=2048,
            )
            log.info("llamaindex_llm_set", provider="huggingface", model=hf_model_id)
            return
        except Exception as exc:
            log.warning("llamaindex_hf_llm_failed", error=str(exc))

    # Gemini fallback
    if google_api_key:
        try:
            from llama_index.llms.google_genai import GoogleGenAI

            Settings.llm = GoogleGenAI(
                model="gemini-2.0-flash",
                api_key=google_api_key,
            )
            log.info("llamaindex_llm_set", provider="gemini", model="gemini-2.0-flash")
            return
        except Exception as exc:
            log.warning("llamaindex_gemini_llm_failed", error=str(exc))

    log.warning("llamaindex_no_llm_configured", hint="Set HF_API_KEY or GOOGLE_API_KEY")
