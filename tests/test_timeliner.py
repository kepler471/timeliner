"""Smoke test – replace with real data tests."""
from datetime import datetime, timedelta

from pathlib import Path
from timeliner.timeliner import Timeliner
from timeliner.retriever import RetrievalMode


def test_empty_timeline():
    builder = Timeliner(retrieval_mode=RetrievalMode.LLM, news_csv=Path("data/news_empty.csv"))  # or .TFIDF / .HYBRID
    start = datetime(2025, 3, 1)
    end = start + timedelta(days=3)
    timeline = builder.build("measles", start, end, timedelta(days=1))
    print(timeline)
    assert len(timeline) == 3
    assert all(entry.summary for entry in timeline)


def test_small_timeline():
    builder = Timeliner(retrieval_mode=RetrievalMode.LLM, news_csv=Path("data/news_headlines.csv"))  # or .TFIDF / .HYBRID
    start = datetime(2025, 3, 1)
    end = start + timedelta(days=3)
    timeline = builder.build("measles", start, end, timedelta(days=1))
    print(timeline)
    assert len(timeline) == 3
    assert all(entry.summary for entry in timeline)