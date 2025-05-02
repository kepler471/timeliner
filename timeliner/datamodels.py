"""Typed representations of raw dataset records."""
from datetime import datetime

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
    # keywords: list[str]


class ExpertSummary(BaseModel):
    id: str
    date: datetime
    theme: str
    description: str
    analysis: str | None = None
    region: str | None = None