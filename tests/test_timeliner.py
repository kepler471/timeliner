"""Smoke test – replace with real data tests."""
from datetime import datetime, timedelta

from timeliner.timeliner import Timeliner
from timeliner.retriever import RetrievalMode


def test_empty_timeline():
    builder = Timeliner(retrieval_mode=RetrievalMode.LLM)  # or .TFIDF / .HYBRID
    start = datetime(2025, 3, 1)
    end = start + timedelta(days=31)
    timeline = builder.build("measles", start, end, timedelta(days=1))
    assert len(timeline) == 31
    assert all(entry.summary for entry in timeline)