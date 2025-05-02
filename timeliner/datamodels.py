"""Typed representations of raw dataset records."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, HttpUrl


class NewsHeadline(BaseModel):
    id: str
    timestamp: datetime
    url: str
    website: str
    country: str
    # headline_original: str
    headline: str
    language: str
    theme: str
    keywords: str
    # keywords: List[str]


class ExpertSummary(BaseModel):
    id: str
    date: datetime
    theme: str
    description: str
    analysis: Optional[str] = None
    region: Optional[str] = None