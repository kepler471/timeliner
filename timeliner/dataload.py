"""Data loaders."""
from __future__ import annotations
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List
import pandas as pd

from .datamodels import NewsHeadline, ExpertSummary
from .logging import get_logger
logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NEWS_CSV = DATA_DIR / "news_headlines.csv"
EXPERT_CSV = DATA_DIR / "expert_summaries.csv"

@lru_cache(maxsize=1)
def _load_news_df() -> pd.DataFrame:
    logger.info("Loading news headlines from %s", NEWS_CSV)
    df = pd.read_csv(NEWS_CSV, parse_dates=["timestamp"], keep_default_na=False)
    df["id"] = df["id"].astype("string")

    # Make timestamps tz-aware, then strip tz to keep them simple & comparable
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert(None).dt.to_pydatetime()

    return df

@lru_cache(maxsize=1)
def _load_expert_df() -> pd.DataFrame:
    logger.info("Loading expert summaries from %s", EXPERT_CSV)
    df = pd.read_csv(EXPERT_CSV, parse_dates=["date"], keep_default_na=False)
    df["id"] = df["id"].astype("string")

    # Make timestamps tz-aware, then strip tz to keep them simple & comparable
    # if df["date"].dt.tz is None:
    #     df["date"] = df["date"].dt.tz_localize("UTC")
    # df["date"] = df["date"].dt.tz_convert(None).dt.to_pydatetime()

    return df


def fetch_headlines(theme: str, start: datetime, end: datetime) -> List[NewsHeadline]:
    df = _load_news_df()
    mask = (df["theme"] == theme) & (df["timestamp"] >= start) & (df["timestamp"] < end)
    return [NewsHeadline.model_validate(rec) for rec in df.loc[mask].to_dict("records")]

def fetch_expert_summaries(theme: str, start: datetime, end: datetime) -> List[ExpertSummary]:
    df = _load_expert_df()
    mask = (df["theme"] == theme) & (df["date"] >= start.date()) & (df["date"] < end.date())
    return [ExpertSummary.model_validate(rec) for rec in df.loc[mask].to_dict("records")]
