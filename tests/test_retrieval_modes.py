"""Tests for different retrieval modes in the timeliner package."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from timeliner.timeliner import Timeliner
from timeliner.retriever import RetrievalMode, ExpertRetriever
from timeliner.summariser import IntervalSummariser


@pytest.fixture
def patched_tfidf_retriever():
    """Patch the TF-IDF retriever to return predictable results."""
    with patch("timeliner.retriever._TFIDFMatcher") as mock_tfidf:
        # Configure the mock to return predetermined IDs
        mock_tfidf.return_value.top_k.return_value = ["e1", "e2"]
        yield mock_tfidf


@pytest.fixture
def patched_llm_retriever():
    """Patch the LLM retriever to return predictable results."""
    with patch("timeliner.retriever._LLMMatcher") as mock_llm:
        # Configure the mock to return predetermined IDs
        mock_llm.return_value.select.return_value = ["e1", "e3"]
        yield mock_llm


class TestRetrievalModes:
    """Test the different retrieval modes."""
    
    def test_tfidf_retrieval_mode(self, sample_expert_summaries, patched_tfidf_retriever):
        """Test the TFIDF retrieval mode."""
        retriever = ExpertRetriever(sample_expert_summaries, mode=RetrievalMode.TFIDF)
        
        summary = "Test summary about measles transmission and vaccination"
        expert_ids = retriever.match(summary)
        
        # Verify results
        assert len(expert_ids) == 2
        assert "e1" in expert_ids
        assert "e2" in expert_ids
        
        # Verify mock was called correctly
        patched_tfidf_retriever.return_value.top_k.assert_called_once_with(summary, k=5)
    
    def test_llm_retrieval_mode(self, sample_expert_summaries, patched_llm_retriever):
        """Test the LLM retrieval mode."""
        retriever = ExpertRetriever(sample_expert_summaries, mode=RetrievalMode.LLM)
        
        summary = "Test summary about measles transmission and public health response"
        expert_ids = retriever.match(summary)
        
        # Verify results
        assert len(expert_ids) == 2
        assert "e1" in expert_ids
        assert "e3" in expert_ids
        
        # Verify mock was called correctly
        patched_llm_retriever.return_value.select.assert_called_once()
    
    def test_hybrid_retrieval_mode(self, sample_expert_summaries, patched_tfidf_retriever, patched_llm_retriever):
        """Test the HYBRID retrieval mode."""
        retriever = ExpertRetriever(sample_expert_summaries, mode=RetrievalMode.HYBRID)
        
        summary = "Test summary about measles transmission and vaccination"
        expert_ids = retriever.match(summary)
        
        # Verify both underlying systems were called
        patched_tfidf_retriever.return_value.top_k.assert_called_once_with(summary, k=5)
        patched_llm_retriever.return_value.select.assert_called_once()
    
    def test_retrieval_mode_in_timeline(self, test_news_csv, test_expert_csv, mock_llm_manager):
        """Test the different retrieval modes in the context of building a timeline."""
        # Mock the ExpertRetriever to track which mode is used
        with patch("timeliner.timeliner.ExpertRetriever") as mock_retriever_cls:
            # Configure mock to return some expert IDs
            mock_instance = MagicMock()
            mock_instance.match.return_value = ["e1", "e2"]
            mock_retriever_cls.return_value = mock_instance
            
            # Create a summarizer that uses our mock LLM
            summarizer = IntervalSummariser(llm=mock_llm_manager)
            
            # Test for each retrieval mode
            for mode in RetrievalMode:
                # Create a timeliner with the current mode
                builder = Timeliner(
                    summariser=summarizer,
                    retrieval_mode=mode,
                    news_csv=test_news_csv,
                    expert_csv=test_expert_csv
                )
                
                # Build a short timeline (1 day)
                start = datetime(2025, 3, 1)
                end = start + timedelta(days=1)
                timeline = builder.build("measles", start, end, timedelta(days=1))
                
                # Verify the timeline
                assert len(timeline) == 1
                assert timeline[0].expert_links == ["e1", "e2"]
                
                # Verify the retriever was constructed with the correct mode
                mock_retriever_cls.assert_called_with(
                    mock_retriever_cls.call_args[0][0],  # expert_summaries arg
                    mode=mode
                )
                
                # Reset the mock for the next iteration
                mock_retriever_cls.reset_mock()
    
    def test_retrieval_performance_comparison(self, sample_expert_summaries):
        """Compare the behavior of different retrieval modes."""
        # Direct test of ExpertRetriever with controlled inputs
        test_summary = "Measles cases rising with concerns about vaccination rates and public health measures."
        
        # Create retrievers for each mode
        tfidf_retriever = ExpertRetriever(sample_expert_summaries, mode=RetrievalMode.TFIDF)
        llm_retriever = ExpertRetriever(sample_expert_summaries, mode=RetrievalMode.LLM)
        hybrid_retriever = ExpertRetriever(sample_expert_summaries, mode=RetrievalMode.HYBRID)
        
        # Mock the underlying components for controlled output
        with patch.object(tfidf_retriever, "tfidf_matcher") as mock_tfidf, \
             patch.object(llm_retriever, "llm_matcher") as mock_llm, \
             patch.object(hybrid_retriever, "tfidf_matcher") as mock_hybrid_tfidf, \
             patch.object(hybrid_retriever, "llm_matcher") as mock_hybrid_llm:
            
            # Configure mocks
            mock_tfidf.top_k.return_value = ["e1", "e2"]
            mock_llm.select.return_value = ["e2", "e3"]
            mock_hybrid_tfidf.top_k.return_value = ["e1", "e2", "e3"]
            mock_hybrid_llm.select.return_value = ["e2"]
            
            # Get results from each retriever
            tfidf_results = tfidf_retriever.match(test_summary)
            llm_results = llm_retriever.match(test_summary)
            hybrid_results = hybrid_retriever.match(test_summary)
            
            # Verify results
            assert set(tfidf_results) == {"e1", "e2"}
            assert set(llm_results) == {"e2", "e3"}
            assert set(hybrid_results) == {"e2"}
            
            # Verify HYBRID mode correctly used TF-IDF to filter before LLM ranking
            filtered_experts = [es for es in sample_expert_summaries if es.id in ["e1", "e2", "e3"]]
            mock_hybrid_llm.select.assert_called_once_with(test_summary, filtered_experts)