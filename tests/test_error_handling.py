"""Tests for error handling and edge cases in the timeliner package."""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path

from timeliner.timeliner import Timeliner
from timeliner.retriever import RetrievalMode, ExpertRetriever, _LLMMatcher
from timeliner.llm_manager import LLMManager
from timeliner.dataload import fetch_headlines, fetch_expert_summaries


class TestErrorHandling:
    """Test error handling in the timeliner package."""
    
    def test_invalid_date_range(self, mock_llm_manager):
        """Test handling of invalid date ranges."""
        builder = Timeliner(
            retrieval_mode=RetrievalMode.TFIDF,
            news_csv=Path("data/news_empty.csv")
        )
        
        start = datetime(2025, 3, 2)
        end = datetime(2025, 3, 1)  # End is before start
        
        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            builder.build("measles", start, end, timedelta(days=1))
        
        assert "`start` must be earlier than `end`" in str(exc_info.value)
    
    def test_nonexistent_csv_file(self):
        """Test handling of non-existent CSV files."""
        nonexistent_file = Path("/path/to/nonexistent/file.csv")
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            fetch_headlines("measles", datetime(2025, 3, 1), datetime(2025, 3, 2), csv_path=nonexistent_file)
    
    def test_malformed_csv_file(self, tmp_path):
        """Test handling of malformed CSV files."""
        # Create a CSV file with missing required columns
        malformed_csv = tmp_path / "malformed.csv"
        with open(malformed_csv, "w") as f:
            f.write("timestamp,headline\n")
            f.write("2025-03-01,Test headline\n")
        
        # Should raise KeyError due to missing columns
        with pytest.raises(KeyError):
            fetch_headlines("measles", datetime(2025, 3, 1), datetime(2025, 3, 2), csv_path=malformed_csv)
    
    def test_llm_failure_handling(self, sample_headlines):
        """Test handling of LLM failures in the summarizer."""
        # Create a mock LLM that raises an exception
        mock_llm = MagicMock(spec=LLMManager)
        mock_llm.chat.side_effect = Exception("LLM service unavailable")
        
        # The summarizer should catch this and return a default message
        from timeliner.summariser import IntervalSummariser
        summarizer = IntervalSummariser(llm=mock_llm)
        
        # Even with an LLM failure, should still return something
        with pytest.raises(Exception):
            summary = summarizer(
                headlines=sample_headlines,
                interval_start=datetime(2025, 3, 1),
                interval_end=datetime(2025, 3, 2),
            )
    
    def test_llm_matcher_failure_handling(self, sample_expert_summaries):
        """Test handling of LLM failures in the matcher."""
        # Create a mock LLM that returns invalid JSON
        mock_llm = MagicMock(spec=LLMManager)
        mock_llm.chat.return_value = "This is not valid JSON"
        
        # The LLM matcher should handle this gracefully
        matcher = _LLMMatcher(mock_llm)
        
        result = matcher.select("Test summary", sample_expert_summaries)
        
        # Should fall back to first 3 briefs
        assert isinstance(result, list)
        assert len(result) <= 3
    
    def test_zero_interval(self, mock_llm_manager):
        """Test handling of zero interval."""
        builder = Timeliner(
            retrieval_mode=RetrievalMode.TFIDF,
            news_csv=Path("data/news_empty.csv")
        )
        
        start = datetime(2025, 3, 1)
        end = datetime(2025, 3, 2)
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            builder.build("measles", start, end, timedelta(days=0))
    
    def test_huge_interval(self, test_news_csv, test_expert_csv, mock_llm_manager):
        """Test handling of intervals larger than the date range."""
        builder = Timeliner(
            retrieval_mode=RetrievalMode.TFIDF,
            news_csv=test_news_csv,
            expert_csv=test_expert_csv
        )
        
        start = datetime(2025, 3, 1)
        end = datetime(2025, 3, 3)
        
        # Interval larger than the date range
        timeline = builder.build("measles", start, end, timedelta(days=10))
        
        # Should still work and produce a single interval
        assert len(timeline) == 1
        assert timeline[0].start == start
        assert timeline[0].end == end
    
    @patch("timeliner.timeliner.fetch_headlines")
    def test_empty_headlines(self, mock_fetch_headlines, mock_llm_manager):
        """Test handling of empty headlines."""
        # Configure mock to return empty list
        mock_fetch_headlines.return_value = []
        
        builder = Timeliner(
            retrieval_mode=RetrievalMode.TFIDF,
            news_csv=Path("data/news_empty.csv")
        )
        
        start = datetime(2025, 3, 1)
        end = start + timedelta(days=1)
        
        timeline = builder.build("measles", start, end, timedelta(days=1))
        
        # Should still work and include an entry with a default summary
        assert len(timeline) == 1
        assert timeline[0].summary == "No headlines for this interval."
    
    def test_edge_case_single_headline(self, test_data_dir, mock_llm_manager):
        """Test with a single headline in the dataset."""
        # Create a dataset with just one headline
        import pandas as pd
        single_headline_csv = test_data_dir / "single_headline.csv"
        
        df = pd.DataFrame({
            "id": ["1"],
            "timestamp": [datetime(2025, 3, 1, 12, 0)],
            "url": ["https://example.com/news/1"],
            "website": ["example.com"],
            "country": ["US"],
            "headline": ["Single measles case reported"],
            "language": ["en"],
            "theme": ["measles"],
            "keywords": ["health,disease,outbreak"]
        })
        
        df.to_csv(single_headline_csv, index=False)
        
        # Create a timeliner with our test data
        builder = Timeliner(
            retrieval_mode=RetrievalMode.TFIDF,
            news_csv=single_headline_csv
        )
        
        start = datetime(2025, 3, 1)
        end = start + timedelta(days=2)
        
        timeline = builder.build("measles", start, end, timedelta(days=1))
        
        # Verify timeline structure
        assert len(timeline) == 2
        assert len(timeline[0].headline_ids) == 1
        assert timeline[0].headline_ids[0] == "1"
        assert len(timeline[1].headline_ids) == 0
    
    def test_different_themes(self, test_news_csv, mock_llm_manager):
        """Test with different themes."""
        builder = Timeliner(
            retrieval_mode=RetrievalMode.TFIDF,
            news_csv=test_news_csv
        )
        
        start = datetime(2025, 3, 1)
        end = start + timedelta(days=1)
        
        # Try with a theme that doesn't exist in the data
        timeline = builder.build("covid", start, end, timedelta(days=1))
        
        # Should still work but have empty headline lists
        assert len(timeline) == 1
        assert len(timeline[0].headline_ids) == 0
        assert timeline[0].summary == "No headlines for this interval."