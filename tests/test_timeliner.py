"""Smoke test – replace with real data tests."""
from datetime import datetime, timedelta

from timeliner.timeliner import Timeliner


def test_empty_timeline():
    builder = Timeliner()
    start = datetime(2025, 1, 1)
    end = start + timedelta(days=3)
    tl = builder.build("test-theme", start, end, timedelta(days=1))
    assert len(tl) == 3
    assert all(entry.summary for entry in tl)