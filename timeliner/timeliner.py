"""Generate a multi‑interval timeline with *cumulative* history passed to the summariser."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Sequence

from pydantic import BaseModel

from .datamodels import ExpertSummary, NewsHeadline
from .dataload import fetch_expert_summaries, fetch_headlines
from .summariser import IntervalSummariser
from .logging import get_logger
from .retriever import ExpertRetriever, RetrievalMode

logger = get_logger(__name__)


class TimelineEntry(BaseModel):
    start: datetime
    end: datetime
    summary: str
    expert_links: List[str]
    headline_ids: List[str]


class Timeliner:
    """Build a timeline where every interval summary knows *all* prior summaries."""

    def __init__(
            self,
            summariser: IntervalSummariser | None = None,
            retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    ):
        self.summariser = summariser or IntervalSummariser()
        self.retrieval_mode = retrieval_mode
    # ------------------------------------------------------------------
    def build(
        self,
        theme: str,
        start: datetime,
        end: datetime,
        interval: timedelta = timedelta(days=1),
    ) -> Sequence[TimelineEntry]:
        if start >= end:
            raise ValueError("`start` must be earlier than `end`.")

        timeline: List[TimelineEntry] = []
        interval_start = start

        # Keep a running list of all previous summaries so far
        cumulative_summaries: List[str] = []
        cumulative_summary: str | None = None

        while interval_start < end:
            interval_end = min(interval_start + interval, end)

            # ------------------------------------------------------------------
            # Fetch data for this interval
            headlines = fetch_headlines(theme, interval_start, interval_end)
            expert_summaries = fetch_expert_summaries(theme, interval_start, interval_end)

            # ------------------------------------------------------------------
            # Call LLM summariser with *cumulative* context
            summary_text = self.summariser(
                headlines=headlines,
                previous_summary=cumulative_summary,
                interval_start=interval_start,
                interval_end=interval_end,
            )

            # ------------------------------------------------------------------
            # Link experts using the chosen retrieval mode
            retriever = ExpertRetriever(expert_summaries, mode=self.retrieval_mode)
            linked_experts = retriever.match(summary_text)
            headline_ids = [h.id for h in headlines]

            # ------------------------------------------------------------------
            # Persist result in timeline & update history
            timeline.append(TimelineEntry(
                start=interval_start,
                end=interval_end,
                summary=summary_text,
                expert_links=linked_experts,
                headline_ids=headline_ids,
            ))

            cumulative_summaries.append(summary_text)
            cumulative_summary = """

""".join(cumulative_summaries)

            interval_start = interval_end

        return timeline

