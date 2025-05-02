"""Basic functional tests for the timeliner package.

These tests are kept simple and only test a few days to minimize LLM compute usage.
More comprehensive tests are available in:
- test_components.py - Tests individual components
- test_timeliner_integration.py - Integration tests with mocked components
- test_retrieval_modes.py - Tests different retrieval modes
- test_error_handling.py - Tests error handling and edge cases
"""
from datetime import datetime, timedelta

from pathlib import Path
from timeliner.timeliner import Timeliner
from timeliner.retriever import RetrievalMode


def test_empty_timeline():
    """Test timeline generation with empty data source."""
    builder = Timeliner(retrieval_mode=RetrievalMode.LLM, news_csv=Path("data/news_empty.csv"))
    start = datetime(2025, 3, 1)
    end = start + timedelta(days=2)  # Keep to 2 days to minimize LLM compute
    timeline = builder.build("measles", start, end, timedelta(days=1))
    
    assert len(timeline) == 2
    assert all(entry.summary for entry in timeline)
    assert all(not entry.headline_ids for entry in timeline)


def test_small_timeline():
    """Test timeline generation with sample data source."""
    builder = Timeliner(retrieval_mode=RetrievalMode.LLM, news_csv=Path("data/news_headlines.csv"))
    start = datetime(2025, 3, 1)
    end = start + timedelta(days=2)  # Keep to 2 days to minimize LLM compute
    timeline = builder.build("measles", start, end, timedelta(days=1))
    
    assert len(timeline) == 2
    assert all(entry.summary for entry in timeline)


def test_tfidf_retrieval():
    """Test timeline generation with TFIDF retrieval mode."""
    builder = Timeliner(retrieval_mode=RetrievalMode.TFIDF, news_csv=Path("data/news_headlines.csv"))
    start = datetime(2025, 3, 1)
    end = start + timedelta(days=2)  # Keep to 2 days to minimize LLM compute
    timeline = builder.build("measles", start, end, timedelta(days=1))
    
    assert len(timeline) == 2
    assert all(entry.summary for entry in timeline)


def test_hybrid_retrieval():
    """Test timeline generation with HYBRID retrieval mode."""
    builder = Timeliner(retrieval_mode=RetrievalMode.HYBRID, news_csv=Path("data/news_headlines.csv"))
    start = datetime(2025, 3, 1)
    end = start + timedelta(days=2)  # Keep to 2 days to minimize LLM compute
    timeline = builder.build("measles", start, end, timedelta(days=1))
    
    assert len(timeline) == 2
    assert all(entry.summary for entry in timeline)