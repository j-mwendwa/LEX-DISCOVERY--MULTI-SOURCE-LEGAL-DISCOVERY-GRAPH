"""
tests/unit/test_knowledge_base.py — Unit tests for the knowledge base search tool.
"""

from src.tools.knowledge_base import mock_qdrant_search


def test_mock_qdrant_search_returns_results():
    """mock_qdrant_search should always return at least 2 results."""
    results = mock_qdrant_search("eviction notice period Kenya")
    assert len(results) >= 2


def test_mock_qdrant_search_result_schema():
    """Each result must match the CaseLawResult TypedDict schema."""
    results = mock_qdrant_search("unlawful eviction")
    for r in results:
        assert "title" in r
        assert "citation" in r
        assert "summary" in r
        assert "relevance_score" in r
        assert isinstance(r["relevance_score"], float)
        assert 0.0 <= r["relevance_score"] <= 1.0


def test_mock_qdrant_search_contains_klr_citations():
    """Results should contain Kenyan Law Reports (KLR) citations."""
    results = mock_qdrant_search("any query")
    citations = [r["citation"] for r in results]
    assert any("KLR" in c or "KEHC" in c for c in citations)
