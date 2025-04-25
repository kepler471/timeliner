# """Configuration loader for timeliner."""
# from __future__ import annotations
# from pydantic import BaseModel, Field
# from datetime import date
# from typing import List, Optional, Dict, Any
# import yaml, pathlib
#
# class TimelinerConfig(BaseModel):
#     theme: str
#     period_start: date
#     period_end: date
#     interval: str = "1D"
#     summary_config_protocols: List[str] = []
#     llm: Dict[str, Any] = Field(default_factory=dict)
#
# def load_config(path: str | pathlib.Path) -> TimelinerConfig:
#     """Read YAML config file into a validated `TimelinerConfig`."""
#     with open(path, "r", encoding="utf-8") as f:
#         data = yaml.safe_load(f)
#     return TimelinerConfig(**data)
#

"""Centralised application settings using Pydantic."""
from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # LLM providers ---------------------------------------------------------
    llm_provider: Literal["openai", "local_llama"] = Field("openai")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    # Timeliner defaults ----------------------------------------------------
    default_period_days: int = 30
    default_interval: Literal["daily", "weekly"] = "daily"

    # Misc ------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:  # noqa: D401
    """Return a cached Settings object."""
    return Settings()