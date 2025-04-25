"""Data‑access stubs – replace with real queries."""
from __future__ import annotations

from datetime import datetime
from typing import List

from .datamodels import ExpertSummary, NewsHeadline


def fetch_headlines(theme: str, start: datetime, end: datetime) -> List[NewsHeadline]:
    return []  # TODO


def fetch_expert_summaries(theme: str, start: datetime, end: datetime) -> List[ExpertSummary]:
    return []  # TODO