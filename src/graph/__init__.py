"""
src/graph/__init__.py — Minimal init to avoid eagerly importing LangGraph/LangChain.
Import graph components directly from their modules when needed.
"""

from src.graph.state import DiscoveryState

__all__ = ["DiscoveryState"]
