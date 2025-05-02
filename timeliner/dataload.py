"""Data loaders."""
from __future__ import annotations
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List
import pandas as pd

from .datamodels import NewsHeadline, ExpertSummary
from .config import get_settings
from .logging import get_logger
logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# Internal helpers                                                   #
# ------------------------------------------------------------------ #
def _resolve_path(default_name: str, override: str | Path | None) -> Path:
    settings = get_settings()
    if override:
        return Path(override)
    # env-driven override in Settings
    explicit = getattr(settings, f"{default_name}_csv", None)
    if explicit:
        return Path(explicit)
    return settings.data_dir / f"{default_name}_headlines.csv" if default_name == "news" \
        else settings.data_dir / f"{default_name}_summaries.csv"


def _preprocess_dates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df[col].dt.tz is None:
        df[col] = df[col].dt.tz_localize("UTC")
    df[col] = df[col].dt.tz_convert(None).dt.to_pydatetime()
    return df


@lru_cache(maxsize=8)
def _load_news_df(csv_path: Path) -> pd.DataFrame:
    logger.info("Loading news headlines from %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], keep_default_na=False)
    
    # Handle empty dataframe
    if df.empty:
        return df
    
    df["id"] = df["id"].astype("string")

    # Make timestamps tz-aware, then strip tz to keep them simple & comparable
    if "timestamp" in df.columns and not df["timestamp"].empty and df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    
    if "timestamp" in df.columns and not df["timestamp"].empty:
        # Convert timestamps to python datetime objects
        import numpy as np
        # Convert to datetime64 first, then to datetime - this avoids the FutureWarning
        datetime_series = df["timestamp"].dt.tz_convert(None)
        df["timestamp"] = pd.Series([pd.Timestamp(ts).to_pydatetime() for ts in datetime_series])

    return df


@lru_cache(maxsize=8)
def _load_expert_df(csv_path: Path) -> pd.DataFrame:
    logger.info("Loading expert summaries from %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["date"], keep_default_na=False)
    
    # Handle empty dataframe
    if df.empty:
        return df
    
    df["id"] = df["id"].astype("string")

    # Make timestamps tz-aware, then strip tz to keep them simple & comparable
    # if df["date"].dt.tz is None:
    #     df["date"] = df["date"].dt.tz_localize("UTC")
    # df["date"] = df["date"].dt.tz_convert(None).dt.to_pydatetime()

    return df


# ------------------------------------------------------------------ #
# Public fetch functions                                             #
# ------------------------------------------------------------------ #
def fetch_headlines(
    theme: str,
    start: datetime,
    end: datetime,
    *,
    csv_path: str | Path | None = None,
) -> List[NewsHeadline]:
    df = _load_news_df(_resolve_path("news", csv_path))
    mask = (df["theme"] == theme) & (df["timestamp"] >= start) & (df["timestamp"] < end)
    return [NewsHeadline.model_validate(rec) for rec in df.loc[mask].to_dict("records")]


def fetch_expert_summaries(
    theme: str,
    start: datetime,
    end: datetime,
    *,
    csv_path: str | Path | None = None,
) -> List[ExpertSummary]:
    df = _load_expert_df(_resolve_path("expert", csv_path))
    
    # Handle empty dataframe
    if df.empty:
        return []
    
    # Convert datetime dates to pandas Timestamps for comparison
    import pandas as pd
    start_date = pd.Timestamp(start.date())
    end_date = pd.Timestamp(end.date())
    
    # Filter with compatible types
    mask = (df["theme"] == theme) & (df["date"] >= start_date) & (df["date"] < end_date)
    return [ExpertSummary.model_validate(rec) for rec in df.loc[mask].to_dict("records")]
