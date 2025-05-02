"""Generate a multi‑interval timeline with *cumulative* history passed to the summariser."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import Sequence, Iterator

from pydantic import BaseModel

from .datamodels import ExpertSummary, NewsHeadline
from .dataload import fetch_expert_summaries, fetch_headlines
from .summariser import IntervalSummariser
from .logging import get_logger
from .retriever import ExpertRetriever, RetrievalMode

logger = get_logger(__name__)


class IntervalData(BaseModel):
    """Data container for a single interval."""
    start: datetime
    end: datetime
    headlines: list[NewsHeadline]
    expert_summaries: list[ExpertSummary]


def generate_intervals(
    start: datetime,
    end: datetime,
    interval: timedelta = timedelta(days=1),
) -> Iterator[tuple[datetime, datetime]]:
    """
    Generate time intervals between start and end.
    
    Args:
        start: Start datetime
        end: End datetime
        interval: Time interval size
        
    Returns:
        Iterator yielding (interval_start, interval_end) tuples
    """
    if start >= end:
        raise ValueError("`start` must be earlier than `end`.")
        
    interval_start = start
    
    while interval_start < end:
        interval_end = min(interval_start + interval, end)
        yield (interval_start, interval_end)
        interval_start = interval_end


class TimelineEntry(BaseModel):
    start: datetime
    end: datetime
    summary: str
    expert_links: list[str]
    headline_ids: list[str]


class Timeliner:
    """Build a timeline where every interval summary knows *all* prior summaries."""

    def __init__(
            self,
            summariser: IntervalSummariser | None = None,
            retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
            news_csv: str | Path | None = None,
            expert_csv: str | Path | None = None,
    ):
        self.summariser = summariser or IntervalSummariser()
        self.retrieval_mode = retrieval_mode
        self.news_csv = news_csv
        self.expert_csv = expert_csv
        
    def fetch_interval_data(
        self, 
        theme: str, 
        interval_start: datetime, 
        interval_end: datetime
    ) -> IntervalData:
        """
        Fetch data for a specific time interval.
        
        Args:
            theme: The theme to filter headlines and expert summaries
            interval_start: Start of the interval
            interval_end: End of the interval
            
        Returns:
            IntervalData containing headlines and expert summaries
        """
        headlines = fetch_headlines(
            theme, 
            interval_start, 
            interval_end, 
            csv_path=self.news_csv
        )
        expert_summaries = fetch_expert_summaries(
            theme, 
            interval_start, 
            interval_end, 
            csv_path=self.expert_csv
        )
        
        return IntervalData(
            start=interval_start,
            end=interval_end,
            headlines=headlines,
            expert_summaries=expert_summaries
        )

    # ------------------------------------------------------------------
    def build(
        self,
        theme: str,
        start: datetime,
        end: datetime,
        interval: timedelta = timedelta(days=1),
    ) -> Sequence[TimelineEntry]:
        """
        Build a timeline for the given theme and date range.
        
        This implementation first builds all intervals, then processes them
        sequentially to create summaries with accumulated context.
        
        Args:
            theme: The theme to filter headlines and expert summaries
            start: Start datetime
            end: End datetime
            interval: Time interval size
            
        Returns:
            Sequence of TimelineEntry objects
        """
        # Generate all intervals upfront
        intervals = list(generate_intervals(start, end, interval))
        
        # Pre-fetch all data for each interval
        interval_data = [
            self.fetch_interval_data(theme, interval_start, interval_end)
            for interval_start, interval_end in intervals
        ]
        
        # Process intervals sequentially with accumulated context
        timeline: list[TimelineEntry] = []
        cumulative_summaries: list[str] = []
        cumulative_summary: str | None = None
        
        for data in interval_data:
            # Call LLM summariser with cumulative context
            summary_text = self.summariser(
                headlines=data.headlines,
                previous_summary=cumulative_summary,
                interval_start=data.start,
                interval_end=data.end,
            )
            
            # Link experts using the chosen retrieval mode
            retriever = ExpertRetriever(data.expert_summaries, mode=self.retrieval_mode)
            linked_experts = retriever.match(summary_text)
            headline_ids = [h.id for h in data.headlines]
            
            # Create timeline entry
            timeline.append(TimelineEntry(
                start=data.start,
                end=data.end,
                summary=summary_text,
                expert_links=linked_experts,
                headline_ids=headline_ids,
            ))
            
            # Update accumulated context
            cumulative_summaries.append(summary_text)
            cumulative_summary = "\n\n".join(cumulative_summaries)
            
        return timeline

