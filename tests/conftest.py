"""Test fixtures for the timeliner package."""
import os
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from unittest.mock import MagicMock

from timeliner.datamodels import NewsHeadline, ExpertSummary
from timeliner.summariser import IntervalSummariser
from timeliner.llm_manager import LLMManager
from timeliner.retriever import ExpertRetriever, RetrievalMode


# Mock data
@pytest.fixture
def sample_headlines() -> List[NewsHeadline]:
    """Return a list of sample news headlines."""
    return [
        NewsHeadline(
            id="1",
            timestamp=datetime(2025, 3, 1, 12, 0),
            url="https://example.com/news/1",
            website="example.com",
            country="US",
            headline="First case of measles confirmed in Texas",
            language="en",
            theme="measles",
            keywords="health,disease,outbreak",
        ),
        NewsHeadline(
            id="2",
            timestamp=datetime(2025, 3, 1, 14, 0),
            url="https://example.com/news/2",
            website="example.com",
            country="US",
            headline="Health officials warn of possible measles exposure in Austin",
            language="en",
            theme="measles",
            keywords="health,disease,exposure",
        ),
        NewsHeadline(
            id="3",
            timestamp=datetime(2025, 3, 2, 9, 0),
            url="https://example.com/news/3",
            website="example.com",
            country="US",
            headline="Three more measles cases confirmed in Texas",
            language="en",
            theme="measles",
            keywords="health,disease,outbreak",
        ),
        NewsHeadline(
            id="4",
            timestamp=datetime(2025, 3, 2, 16, 0),
            url="https://example.com/news/4",
            website="example.com",
            country="US",
            headline="CDC monitoring Texas measles outbreak",
            language="en",
            theme="measles",
            keywords="health,disease,CDC",
        ),
    ]


@pytest.fixture
def sample_expert_summaries() -> List[ExpertSummary]:
    """Return a list of sample expert summaries."""
    return [
        ExpertSummary(
            id="e1",
            date=datetime(2025, 3, 1),
            theme="measles",
            description="Analysis of measles transmission patterns",
            analysis="Measles is highly contagious with an R0 of 12-18, meaning each infected person can infect 12-18 others in a susceptible population.",
            region="Global",
        ),
        ExpertSummary(
            id="e2",
            date=datetime(2025, 3, 1),
            theme="measles",
            description="Vaccination efficacy against measles",
            analysis="Two doses of MMR vaccine are approximately 97% effective at preventing measles infection.",
            region="Global",
        ),
        ExpertSummary(
            id="e3",
            date=datetime(2025, 3, 2),
            theme="measles",
            description="Public health response to outbreaks",
            analysis="Contact tracing and quarantine measures are essential to contain measles outbreaks, alongside vaccination campaigns.",
            region="US",
        ),
    ]


# Mocked components
@pytest.fixture
def mock_llm_manager() -> MagicMock:
    """Return a mocked LLMManager that returns predefined responses."""
    mock = MagicMock(spec=LLMManager)
    
    # Configure the chat method to return different responses for different inputs
    def mock_chat(messages):
        # Very simplified response logic - in real tests you might want to match on message content
        system_message = messages[0].content if messages else ""
        if "analyst generating concise timeline summaries" in system_message:
            return "Day 1: First measles case confirmed in Texas with potential exposure in Austin."
        elif "precise assistant that outputs JSON" in system_message:
            return '["e1", "e2"]'
        return "Default mocked response"
    
    mock.chat.side_effect = mock_chat
    return mock


@pytest.fixture
def mock_interval_summariser(mock_llm_manager) -> IntervalSummariser:
    """Return a mocked IntervalSummariser."""
    return IntervalSummariser(llm=mock_llm_manager)


# Test data files
@pytest.fixture
def test_data_dir(tmp_path) -> Path:
    """Create and return a temporary directory with test data files."""
    # Create test headlines CSV
    headlines_df = pd.DataFrame({
        "id": ["1", "2", "3", "4"],
        "timestamp": [
            datetime(2025, 3, 1, 12, 0),
            datetime(2025, 3, 1, 14, 0),
            datetime(2025, 3, 2, 9, 0),
            datetime(2025, 3, 2, 16, 0)
        ],
        "url": [f"https://example.com/news/{i}" for i in range(1, 5)],
        "website": ["example.com"] * 4,
        "country": ["US"] * 4,
        "headline": [
            "First case of measles confirmed in Texas",
            "Health officials warn of possible measles exposure in Austin",
            "Three more measles cases confirmed in Texas",
            "CDC monitoring Texas measles outbreak"
        ],
        "language": ["en"] * 4,
        "theme": ["measles"] * 4,
        "keywords": ["health,disease,outbreak", "health,disease,exposure", 
                     "health,disease,outbreak", "health,disease,CDC"]
    })
    
    # Create test expert summaries CSV
    expert_df = pd.DataFrame({
        "id": ["e1", "e2", "e3"],
        "date": [
            datetime(2025, 3, 1),
            datetime(2025, 3, 1),
            datetime(2025, 3, 2)
        ],
        "theme": ["measles"] * 3,
        "description": [
            "Analysis of measles transmission patterns",
            "Vaccination efficacy against measles",
            "Public health response to outbreaks"
        ],
        "analysis": [
            "Measles is highly contagious with an R0 of 12-18, meaning each infected person can infect 12-18 others in a susceptible population.",
            "Two doses of MMR vaccine are approximately 97% effective at preventing measles infection.",
            "Contact tracing and quarantine measures are essential to contain measles outbreaks, alongside vaccination campaigns."
        ],
        "region": ["Global", "Global", "US"]
    })
    
    # Save to temporary directory
    headlines_path = tmp_path / "test_news_headlines.csv"
    expert_path = tmp_path / "test_expert_summaries.csv"
    
    headlines_df.to_csv(headlines_path, index=False)
    expert_df.to_csv(expert_path, index=False)
    
    return tmp_path


@pytest.fixture
def test_news_csv(test_data_dir) -> Path:
    """Return the path to the test news CSV file."""
    return test_data_dir / "test_news_headlines.csv"


@pytest.fixture
def test_expert_csv(test_data_dir) -> Path:
    """Return the path to the test expert summaries CSV file."""
    return test_data_dir / "test_expert_summaries.csv"