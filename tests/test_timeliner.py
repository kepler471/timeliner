"""Basic functional tests for the timeliner package.

These tests are kept simple and only test a few days to minimize LLM compute usage.
All tests display the generated timeline for quality assessment.

More comprehensive tests available in:
- test_components.py - Tests individual components
- test_timeliner_integration.py - Integration tests with mocked components
- test_retrieval_modes.py - Tests different retrieval modes
- test_error_handling.py - Tests error handling and edge cases
"""
from datetime import datetime, timedelta

from pathlib import Path
from timeliner.timeliner import Timeliner, generate_intervals
from timeliner.retriever import RetrievalMode


def print_timeline(timeline):
    """Helper to print timeline entries in a readable format."""
    print("\n" + "="*80)
    print(f"Timeline with {len(timeline)} entries:")
    print("="*80)
    
    for i, entry in enumerate(timeline):
        print(f"\nDAY {i+1}: {entry.start.date()} to {entry.end.date()}")
        print("-"*80)
        print(f"Summary: {entry.summary}")
        print(f"Expert links: {entry.expert_links}")
        print(f"Headlines [{len(entry.headline_ids)}]: {entry.headline_ids[:5]}{'...' if len(entry.headline_ids) > 5 else ''}")
    
    print("\n" + "="*80)
    return timeline


def test_empty_timeline():
    """Test timeline generation with empty data source."""
    builder = Timeliner(retrieval_mode=RetrievalMode.LLM, news_csv=Path("data/news_empty.csv"))
    start = datetime(2025, 3, 1)
    end = start + timedelta(days=2)  # Keep to 2 days to minimize LLM compute
    timeline = builder.build("measles", start, end, timedelta(days=1))

    # Display results for assessment
    print_timeline(timeline)

    assert len(timeline) == 2
    assert all(entry.summary for entry in timeline)
    assert all(not entry.headline_ids for entry in timeline)


def test_small_timeline():
    """Test timeline generation with sample data source."""
    builder = Timeliner(retrieval_mode=RetrievalMode.LLM, news_csv=Path("data/news_headlines.csv"))
    start = datetime(2025, 3, 1)
    end = start + timedelta(days=2)  # Keep to 2 days to minimize LLM compute
    timeline = builder.build("measles", start, end, timedelta(days=1))

    # Display results for assessment
    print_timeline(timeline)

    assert len(timeline) == 2
    assert all(entry.summary for entry in timeline)


# def test_tfidf_retrieval():
#     """Test timeline generation with TFIDF retrieval mode."""
#     builder = Timeliner(retrieval_mode=RetrievalMode.TFIDF, news_csv=Path("data/news_headlines.csv"))
#     start = datetime(2025, 3, 1)
#     end = start + timedelta(days=2)  # Keep to 2 days to minimize LLM compute
#     timeline = builder.build("measles", start, end, timedelta(days=1))
#
#     # Display results for assessment
#     print_timeline(timeline)
#
#     assert len(timeline) == 2
#     assert all(entry.summary for entry in timeline)
#
#
# def test_hybrid_retrieval():
#     """Test timeline generation with HYBRID retrieval mode."""
#     builder = Timeliner(retrieval_mode=RetrievalMode.HYBRID, news_csv=Path("data/news_headlines.csv"))
#     start = datetime(2025, 3, 1)
#     end = start + timedelta(days=2)  # Keep to 2 days to minimize LLM compute
#     timeline = builder.build("measles", start, end, timedelta(days=1))
#
#     # Display results for assessment
#     print_timeline(timeline)
#
#     assert len(timeline) == 2
#     assert all(entry.summary for entry in timeline)


def test_generate_intervals():
    """Test the interval generator function."""
    # Test with exact number of days
    start = datetime(2025, 3, 1)
    end = datetime(2025, 3, 5)
    interval = timedelta(days=1)
    
    intervals = list(generate_intervals(start, end, interval))
    
    assert len(intervals) == 4
    assert intervals[0] == (datetime(2025, 3, 1), datetime(2025, 3, 2))
    assert intervals[1] == (datetime(2025, 3, 2), datetime(2025, 3, 3))
    assert intervals[2] == (datetime(2025, 3, 3), datetime(2025, 3, 4))
    assert intervals[3] == (datetime(2025, 3, 4), datetime(2025, 3, 5))
    
    # Test with non-standard interval
    start = datetime(2025, 3, 1)
    end = datetime(2025, 3, 5)
    interval = timedelta(days=2)
    
    intervals = list(generate_intervals(start, end, interval))
    
    assert len(intervals) == 2
    assert intervals[0] == (datetime(2025, 3, 1), datetime(2025, 3, 3))
    assert intervals[1] == (datetime(2025, 3, 3), datetime(2025, 3, 5))
    
    # Test with partial day at the end
    start = datetime(2025, 3, 1)
    end = datetime(2025, 3, 4, 12)  # End at noon on March 4
    interval = timedelta(days=1)
    
    intervals = list(generate_intervals(start, end, interval))
    
    assert len(intervals) == 4
    assert intervals[0] == (datetime(2025, 3, 1), datetime(2025, 3, 2))
    assert intervals[1] == (datetime(2025, 3, 2), datetime(2025, 3, 3))
    assert intervals[2] == (datetime(2025, 3, 3), datetime(2025, 3, 4))
    assert intervals[3] == (datetime(2025, 3, 4), datetime(2025, 3, 4, 12))


def test_get_cumulative_summary():
    """Test extracting cumulative summary from timeline entries."""
    from timeliner.timeliner import TimelineEntry
    
    # Create sample timeline entries
    entries = [
        TimelineEntry(
            start=datetime(2025, 3, 1),
            end=datetime(2025, 3, 2),
            summary="Day 1 summary",
            expert_links=["e1"],
            headline_ids=["1", "2"],
        ),
        TimelineEntry(
            start=datetime(2025, 3, 2),
            end=datetime(2025, 3, 3),
            summary="Day 2 summary",
            expert_links=["e2"],
            headline_ids=["3", "4"],
        ),
    ]
    
    # Test getting summary from timeline
    timeliner = Timeliner()
    summary = timeliner.get_cumulative_summary(entries)
    
    assert summary == "Day 1 summary\n\nDay 2 summary"
    
    # Test with empty timeline
    summary = timeliner.get_cumulative_summary([])
    assert summary is None


def test_timeline_continuation():
    """Test continuation of an existing timeline."""
    # Create initial timeline for the first 2 days
    builder = Timeliner(retrieval_mode=RetrievalMode.LLM, news_csv=Path("data/news_headlines.csv"))
    start = datetime(2025, 3, 1)
    initial_end = datetime(2025, 3, 3)  # 2 days
    
    # Build the initial timeline
    initial_timeline = builder.build("measles", start, initial_end, timedelta(days=1))
    
    # Verify the initial timeline
    assert len(initial_timeline) == 2
    assert initial_timeline[0].start == start
    assert initial_timeline[1].end == initial_end
    
    # Continue the timeline for 1 more day
    continuation_end = datetime(2025, 3, 4)  # Add 1 more day
    
    # Approach 1: Use the continue_timeline method directly
    extended_timeline1 = builder.continue_timeline(
        initial_timeline, 
        "measles", 
        continuation_end, 
        timedelta(days=1)
    )
    
    # Approach 2: Use the build method with existing_timeline
    extended_timeline2 = builder.build(
        "measles",
        start,  # Original start date, should be ignored for continuation
        continuation_end,
        timedelta(days=1),
        existing_timeline=initial_timeline
    )
    
    # Display results for assessment
    print_timeline(extended_timeline1)
    
    # Verify the extended timeline
    assert len(extended_timeline1) == 3  # Initial 2 days + 1 more day
    assert extended_timeline1[0].start == start
    assert extended_timeline1[2].end == continuation_end
    
    # Check that both approaches yield the same result
    assert len(extended_timeline1) == len(extended_timeline2)
    assert extended_timeline1[0].start == extended_timeline2[0].start
    assert extended_timeline1[-1].end == extended_timeline2[-1].end