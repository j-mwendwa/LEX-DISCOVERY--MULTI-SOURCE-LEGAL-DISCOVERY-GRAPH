"""
src/core/llm_factory.py — LLM factory for LEX-DISCOVERY.

Priority order:
  1. HuggingFace Saul-7B (law-specialised) via Inference API
  2. Google Gemini (fallback — fast, reliable)

Usage:
    from src.core.llm_factory import get_llm, get_law_llm
    llm = get_llm()           # Returns best available LLM
    llm = get_law_llm()       # Always returns law-tuned model with fallback
"""
from __future__ import annotations

from src.config import cfg, settings
from src.core.logging import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace — Saul Legal AI (primary)
# ─────────────────────────────────────────────────────────────────────────────
def _build_hf_llm(
    model_id: str = "Equall/Saul-7B-Instruct-v1",
    temperature: float = 0.1,
    max_new_tokens: int = 2048,
    api_key: str = "",
):
    """
    Build a LangChain HuggingFaceEndpoint for Saul Legal AI.
    Requires: pip install langchain-huggingface
    """
    try:
        from langchain_huggingface import HuggingFaceEndpoint

        token = api_key or settings.hf_api_key
        if not token:
            raise ValueError("HF_API_KEY not set")

        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            huggingfacehub_api_token=token,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            task="text-generation",
        )
        log.info("hf_llm_built", model=model_id)
        return llm
    except Exception as exc:
        log.warning("hf_llm_build_failed", model=model_id, error=str(exc))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Google Gemini — Fallback
# ─────────────────────────────────────────────────────────────────────────────
def _build_gemini_llm(
    model: str = "gemini-2.0-flash",
    temperature: float = 0.0,
):
    """Build LangChain ChatGoogleGenerativeAI as fallback."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=settings.google_api_key or None,
        )
        log.info("gemini_llm_built", model=model)
        return llm
    except Exception as exc:
        log.error("gemini_llm_build_failed", error=str(exc))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LangSmith callbacks
# ─────────────────────────────────────────────────────────────────────────────
def get_langsmith_callbacks(run_name: str = "lex-discovery") -> list:
    """Return LangSmith tracer callbacks if configured."""
    if not settings.langsmith_api_key:
        return []
    try:
        from langchain.callbacks.tracers import LangChainTracer
        from langsmith import Client

        tracer = LangChainTracer(
            project_name=settings.langsmith_project,
            client=Client(
                api_url=settings.langchain_endpoint,
                api_key=settings.langsmith_api_key,
            ),
        )
        return [tracer]
    except Exception as exc:
        log.warning("langsmith_callbacks_failed", error=str(exc))
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────
def get_llm(temperature: float | None = None, use_law_model: bool = True):
    """
    Return the best available LLM.

    Strategy:
      1. Try HuggingFace Saul-7B (if HF_API_KEY set and use_law_model=True)
      2. Fall back to Gemini gemini-2.0-flash

    Args:
        temperature: Override temperature. Defaults to config value.
        use_law_model: If True, prefer the law-specialised HF model.

    Returns:
        A LangChain-compatible LLM instance.
    """
    temp = temperature if temperature is not None else cfg.get("llm", {}).get("temperature", 0.1)
    model_id = settings.hf_model_id or cfg.get("llm", {}).get("default_model", "Equall/Saul-7B-Instruct-v1")
    max_new_tokens = cfg.get("llm", {}).get("max_new_tokens", 2048)

    if use_law_model and settings.hf_api_key:
        llm = _build_hf_llm(model_id=model_id, temperature=temp, max_new_tokens=max_new_tokens)
        if llm is not None:
            return llm

    # Gemini fallback
    fallback = cfg.get("llm", {}).get("fallback_model", "gemini-2.0-flash")
    llm = _build_gemini_llm(model=fallback, temperature=temp)
    if llm is not None:
        return llm

    raise RuntimeError(
        "No LLM available. Set HF_API_KEY (for Saul-7B) or GOOGLE_API_KEY (for Gemini)."
    )


def get_law_llm(temperature: float = 0.1):
    """Convenience wrapper: always returns the law-specialised LLM."""
    return get_llm(temperature=temperature, use_law_model=True)
