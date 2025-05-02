"""Integration tests for the timeliner package."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from timeliner.timeliner import Timeliner, TimelineEntry
from timeliner.retriever import RetrievalMode
from timeliner.summariser import IntervalSummariser


class TestTimelineBuilder:
    """Integration tests for the Timeliner class."""
    
    def test_build_timeline_mock_llm(self, test_news_csv, test_expert_csv, mock_llm_manager):
        """Test building a timeline with a mocked LLM."""
        # Create a summarizer that uses our mock LLM
        summarizer = IntervalSummariser(llm=mock_llm_manager)
        
        # Create a timeliner with our mocked components
        builder = Timeliner(
            summariser=summarizer,
            retrieval_mode=RetrievalMode.LLM,
            news_csv=test_news_csv,
            expert_csv=test_expert_csv
        )
        
        # Build a short timeline (2 days)
        start = datetime(2025, 3, 1)
        end = start + timedelta(days=2)
        timeline = builder.build("measles", start, end, timedelta(days=1))
        
        # Verify the timeline structure
        assert len(timeline) == 2
        assert all(isinstance(entry, TimelineEntry) for entry in timeline)
        assert all(entry.summary for entry in timeline)
        
        # Check first day
        assert timeline[0].start == start
        assert timeline[0].end == start + timedelta(days=1)
        
        # Check second day
        assert timeline[1].start == start + timedelta(days=1)
        assert timeline[1].end == end
    
    def test_build_timeline_with_tfidf_retrieval(self, test_news_csv, test_expert_csv, mock_llm_manager):
        """Test building a timeline with TF-IDF retrieval."""
        # Create a summarizer that uses our mock LLM
        summarizer = IntervalSummariser(llm=mock_llm_manager)
        
        # Create a timeliner with TF-IDF retrieval
        builder = Timeliner(
            summariser=summarizer,
            retrieval_mode=RetrievalMode.TFIDF,
            news_csv=test_news_csv,
            expert_csv=test_expert_csv
        )
        
        # Build a short timeline (2 days)
        start = datetime(2025, 3, 1)
        end = start + timedelta(days=2)
        timeline = builder.build("measles", start, end, timedelta(days=1))
        
        # Verify the timeline structure
        assert len(timeline) == 2
        assert all(isinstance(entry, TimelineEntry) for entry in timeline)
        assert all(entry.summary for entry in timeline)
    
    def test_build_timeline_with_empty_data(self, test_data_dir, mock_llm_manager):
        """Test building a timeline with empty data sources."""
        # Create empty CSV files
        import pandas as pd
        empty_news = test_data_dir / "empty_news.csv"
        empty_experts = test_data_dir / "empty_experts.csv"
        
        # Create empty dataframes with correct columns
        news_df = pd.DataFrame(columns=[
            "id", "timestamp", "url", "website", "country", 
            "headline", "language", "theme", "keywords"
        ])
        expert_df = pd.DataFrame(columns=[
            "id", "date", "theme", "description", "analysis", "region"
        ])
        
        news_df.to_csv(empty_news, index=False)
        expert_df.to_csv(empty_experts, index=False)
        
        # Create a summarizer that uses our mock LLM
        summarizer = IntervalSummariser(llm=mock_llm_manager)
        
        # Create a timeliner
        builder = Timeliner(
            summariser=summarizer,
            retrieval_mode=RetrievalMode.HYBRID,
            news_csv=empty_news,
            expert_csv=empty_experts
        )
        
        # Build a short timeline (2 days)
        start = datetime(2025, 3, 1)
        end = start + timedelta(days=2)
        timeline = builder.build("measles", start, end, timedelta(days=1))
        
        # Verify the timeline structure
        assert len(timeline) == 2
        assert all(isinstance(entry, TimelineEntry) for entry in timeline)
        assert all(entry.summary for entry in timeline)
        assert all(not entry.expert_links for entry in timeline)
        assert all(not entry.headline_ids for entry in timeline)
    
    @patch("timeliner.timeliner.fetch_headlines")
    @patch("timeliner.timeliner.fetch_expert_summaries")
    def test_build_timeline_data_flow(self, mock_fetch_experts, mock_fetch_headlines, 
                                     mock_llm_manager, sample_headlines, sample_expert_summaries):
        """Test the data flow through the timeline builder."""
        # Set up our mocks to return test data
        mock_fetch_headlines.return_value = sample_headlines[:2]  # First day headlines
        mock_fetch_experts.return_value = sample_expert_summaries[:2]  # First day experts
        
        # Create a summarizer that uses our mock LLM
        summarizer = IntervalSummariser(llm=mock_llm_manager)
        
        # Create a timeliner
        builder = Timeliner(
            summariser=summarizer,
            retrieval_mode=RetrievalMode.LLM,
        )
        
        # Build a short timeline (1 day only)
        start = datetime(2025, 3, 1)
        end = start + timedelta(days=1)
        timeline = builder.build("measles", start, end, timedelta(days=1))
        
        # Verify the timeline structure
        assert len(timeline) == 1
        assert timeline[0].headline_ids == ["1", "2"]  # IDs from our mock headlines
        assert isinstance(timeline[0].expert_links, list)
        
        # Verify our mocks were called correctly
        mock_fetch_headlines.assert_called_once()
        mock_fetch_experts.assert_called_once()
        
    def test_build_timeline_with_custom_interval(self, test_news_csv, test_expert_csv, mock_llm_manager):
        """Test building a timeline with a custom interval."""
        # Create a summarizer that uses our mock LLM
        summarizer = IntervalSummariser(llm=mock_llm_manager)
        
        # Create a timeliner
        builder = Timeliner(
            summariser=summarizer,
            retrieval_mode=RetrievalMode.HYBRID,
            news_csv=test_news_csv,
            expert_csv=test_expert_csv
        )
        
        # Build a timeline with 12-hour intervals (3 intervals over 36 hours)
        start = datetime(2025, 3, 1)
        end = start + timedelta(hours=36)
        timeline = builder.build("measles", start, end, timedelta(hours=12))
        
        # Verify the timeline structure
        assert len(timeline) == 3
        
        # Check intervals
        assert timeline[0].start == start
        assert timeline[0].end == start + timedelta(hours=12)
        
        assert timeline[1].start == start + timedelta(hours=12)
        assert timeline[1].end == start + timedelta(hours=24)
        
        assert timeline[2].start == start + timedelta(hours=24)
        assert timeline[2].end == end