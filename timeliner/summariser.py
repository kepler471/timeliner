"""LLM‑powered summarisation for a single interval."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from langchain.schema import HumanMessage, SystemMessage  # type: ignore

from .datamodels import NewsHeadline
from .llm_manager import LLMManager
from .protocols import summary_config_protocols
from .logging import get_logger

logger = get_logger(__name__)


class IntervalSummariser:
    def __init__(self, llm: LLMManager | None = None):
        self.llm = llm or LLMManager()

    # ------------------------------------------------------------------
    def __call__(
        self,
        headlines: Sequence[NewsHeadline],
        previous_summary: str | None = None,
        interval_start: datetime | None = None,
        interval_end: datetime | None = None,
    ) -> str:
        if not headlines:
            return "No headlines for this interval."

        system_prompt = "\n".join([
            "You are an analyst generating concise timeline summaries from news headlines.",
            *summary_config_protocols,
        ])

        bullet_list = "\n".join(f"- {h.headline}" for h in headlines)
        human_parts = [
            f"Time interval: {interval_start} – {interval_end} (ISO)",
            "Headlines:",
            bullet_list,
        ]
        if previous_summary:
            human_parts.extend([
                "Previous summary:",
                previous_summary,
                "Focus on *new* information compared to previous summary.",
            ])
        human_prompt = "\n".join(human_parts)

        logger.debug("System prompt:\n%s", system_prompt)
        logger.debug("Human prompt:\n%s", human_prompt)

        return self.llm.chat([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])