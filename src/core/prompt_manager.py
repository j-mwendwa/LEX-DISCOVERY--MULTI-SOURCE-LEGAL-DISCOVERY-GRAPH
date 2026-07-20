"""
src/core/prompt_manager.py — Load and version prompts from the prompts/ directory.

Usage:
    from src.core.prompt_manager import load_prompt
    system_prompt = load_prompt("lead_attorney", "v1")
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.core.exceptions import ConfigError
from src.core.logging import get_logger

log = get_logger(__name__)
_PROMPTS_ROOT = Path(__file__).parent.parent.parent / "prompts"


@lru_cache(maxsize=64)
def load_prompt(name: str, version: str = "v1") -> str:
    """
    Load a prompt from prompts/<name>_<version>.md or prompts/<subdir>/<name>_<version>.md.

    Args:
        name: Prompt name (e.g. 'lead_attorney', 'timeline_extractor').
        version: Version string (e.g. 'v1', 'v2').

    Returns:
        The prompt text as a string.

    Raises:
        ConfigError: If the prompt file does not exist.
    """
    candidates = [
        _PROMPTS_ROOT / f"{name}_{version}.md",
        _PROMPTS_ROOT / "system" / f"{name}_{version}.md",
        _PROMPTS_ROOT / name / f"{version}.md",
    ]
    for path in candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            log.debug("prompt_loaded", name=name, version=version, path=str(path))
            return text

    raise ConfigError(
        f"Prompt '{name}' version '{version}' not found. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def reload_prompts() -> None:
    """Invalidate the prompt cache (call on hot-reload)."""
    load_prompt.cache_clear()
    log.info("prompt_cache_cleared")
