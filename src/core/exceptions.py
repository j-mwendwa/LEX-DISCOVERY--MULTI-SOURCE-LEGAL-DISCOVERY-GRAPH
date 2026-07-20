"""
src/core/exceptions.py — Domain-specific exceptions for LEX-DISCOVERY.
"""


class LexDiscoveryError(Exception):
    """Base exception for all LEX-DISCOVERY errors."""


class IngestionError(LexDiscoveryError):
    """Raised when PDF ingestion or parsing fails."""


class ExtractionError(LexDiscoveryError):
    """Raised when LLM-based metadata/timeline extraction fails."""


class SearchError(LexDiscoveryError):
    """Raised when Qdrant search fails."""


class ConfigError(LexDiscoveryError):
    """Raised when a configuration value is missing or invalid."""


class AuthError(LexDiscoveryError):
    """Raised on invalid API key authentication."""


class HITLError(LexDiscoveryError):
    """Raised when Human-in-the-Loop state is invalid."""
