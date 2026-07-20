"""
tests/unit/test_qdrant_store.py — Unit tests for the Qdrant vector store helper.
"""
from src.vectordb.qdrant_store import _bm25_encode


def test_bm25_encode_empty_string():
    """BM25 encoder should return empty indices/values for empty input."""
    indices, values = _bm25_encode("")
    assert indices == []
    assert values == []


def test_bm25_encode_single_word():
    """BM25 encoder should produce one entry for a single token."""
    indices, values = _bm25_encode("eviction")
    assert len(indices) == 1
    assert len(values) == 1
    assert all(isinstance(i, int) for i in indices)
    assert all(isinstance(v, float) for v in values)


def test_bm25_encode_multiple_words():
    """BM25 encoder should produce an entry per unique token."""
    indices, values = _bm25_encode("eviction notice kenya")
    assert len(indices) == 3
    assert len(values) == 3


def test_bm25_encode_is_deterministic():
    """Same input always produces same output."""
    text = "landlord tenant eviction notice period"
    result1 = _bm25_encode(text)
    result2 = _bm25_encode(text)
    assert result1 == result2


def test_bm25_values_are_positive():
    """All BM25 values should be positive floats."""
    _, values = _bm25_encode("notice period thirty days")
    assert all(v > 0.0 for v in values)
