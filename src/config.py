"""
src/config.py — Unified settings and config loader for LEX-DISCOVERY.

Usage everywhere:
    from src.config import settings, cfg
    # settings -> pydantic Settings (typed, validated, from .env)
    # cfg      -> dict from configs/config.yaml
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

import yaml
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_FILE = _PROJECT_ROOT / "configs" / "config.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Settings — loaded from .env / environment
# ─────────────────────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    # AI — HuggingFace (primary law LLM)
    hf_api_key: str = ""
    hf_model_id: str = "Equall/Saul-7B-Instruct-v1"

    # AI — Google Gemini (fallback)
    google_api_key: str = ""

    # Vector store — Qdrant Cloud
    qdrant_url: str = "https://your-cluster.qdrant.tech"
    qdrant_api_key: str = ""
    vector_backend: str = "qdrant"

    # Auth
    allowed_api_keys: List[str] = ["dev-local-key"]
    app_env: str = "development"

    # Memory
    memory_encryption_key: str = ""

    # LangSmith Observability
    langsmith_api_key: str = ""
    langsmith_project: str = "lex-discovery"
    langchain_tracing_v2: str = "false"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @model_validator(mode="after")
    def _reject_default_key_in_production(self) -> "Settings":
        if self.app_env == "production" and "dev-local-key" in self.allowed_api_keys:
            raise ValueError(
                "Production mode detected but default API key 'dev-local-key' is still set. "
                "Set ALLOWED_API_KEYS to real keys before deploying."
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# YAML config loader
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_yaml() -> dict:
    if not _CONFIG_FILE.exists():
        return {}
    with _CONFIG_FILE.open() as fh:
        return yaml.safe_load(fh) or {}


# Public singletons
@lru_cache(maxsize=1)
def _get_settings() -> Settings:
    return Settings()


settings: Settings = _get_settings()
cfg: dict = _load_yaml()
