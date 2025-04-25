"""
Expert-summary retrieval helpers for Timeliner.

Modes
=====
TFIDF   – deterministic, cheap word-overlap match
LLM     – direct LLM reasoning over *all* briefs
HYBRID  – TF-IDF shortlist -> LLM re-rank (recommended)
"""
from __future__ import annotations

import json
from enum import Enum
from functools import lru_cache
from typing import List, Sequence

import numpy as np
from langchain.schema import HumanMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .datamodels import ExpertSummary
from .llm_manager import LLMManager
from .logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Retrieval modes                                                             #
# --------------------------------------------------------------------------- #
class RetrievalMode(str, Enum):
    TFIDF = "tfidf"
    LLM = "llm"
    HYBRID = "hybrid"


# --------------------------------------------------------------------------- #
# TF-IDF helper (stage-A fast filter)                                         #
# --------------------------------------------------------------------------- #
class _TFIDFMatcher:
    def __init__(self, expert_summaries: Sequence[ExpertSummary]):
        self.ids: List[str] = [es.id for es in expert_summaries]
        docs = [
            f"{es.description or ''} {es.analysis or ''}".strip()
            for es in expert_summaries
        ]
        self.vectorizer = _get_vectorizer()
        self.matrix = self.vectorizer.fit_transform(docs)

    def top_k(self, query: str, k: int = 5) -> List[ExpertSummary]:
        if not query.strip():
            return []
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).ravel()
        idxs = np.argsort(scores)[-k:][::-1]
        return [self.ids[i] for i in idxs if scores[i] > 0.1]


@lru_cache(maxsize=4)
def _get_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=4096,
    )


# --------------------------------------------------------------------------- #
# LLM re-rank / filtering                                                     #
# --------------------------------------------------------------------------- #
class _LLMMatcher:
    """Ask the LLM which briefs matter."""

    def __init__(self, llm: LLMManager | None = None):
        self.llm = llm or LLMManager(provider="ollama")  # default to local

    def _build_prompt(self, summary: str, briefs: Sequence[ExpertSummary]) -> List:
        # truncate each brief to ~120 words
        def _snippet(es: ExpertSummary) -> str:
            text = f"{es.description or ''} {es.analysis or ''}".strip()
            return " ".join(text.split()[:120])

        buf: list[str] = [
            "You link daily timeline summaries with expert briefs.",
            "Return **only** a JSON array of IDs that enrich the summary.",
            "",
            "Interval summary:",
            summary,
            "",
            "Candidate briefs:",
        ]
        for es in briefs:
            buf.extend(
                [
                    f"ID = {es.id}",
                    '"""',
                    _snippet(es),
                    '"""',
                    "",
                ]
            )

        system_msg = SystemMessage(
            content="You are a precise assistant that outputs JSON only."
        )
        human_msg = HumanMessage(content="\n".join(buf))
        return [system_msg, human_msg]

    def select(self, summary: str, briefs: Sequence[ExpertSummary]) -> List[str]:
        if not briefs:
            return []
        messages = self._build_prompt(summary, briefs)
        try:
            raw = self.llm.chat(messages)
            ids = json.loads(raw)
            return [i for i in ids if isinstance(i, str)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM select failed (%s); falling back to first 3 briefs", exc)
            return [b.id for b in briefs[:3]]


class ExpertRetriever:
    """Factory wrapper exposing `.match(summary_text) -> List[str]`."""

    def __init__(
        self,
        expert_summaries: Sequence[ExpertSummary],
        mode: RetrievalMode = RetrievalMode.HYBRID,
        llm: LLMManager | None = None,
    ):
        self.mode = mode
        self.expert_summaries = list(expert_summaries)

        # Exit if there is nothing to index
        if not self.expert_summaries:
            self.llm_matcher = None
            self.tfidf_matcher = None
            return                      # keep attributes but skip heavy init

        self.llm_matcher = _LLMMatcher(llm) if mode in (RetrievalMode.LLM, RetrievalMode.HYBRID) else None
        self.tfidf_matcher = _TFIDFMatcher(expert_summaries) if mode != RetrievalMode.LLM else None

    # ------------------------------------------------------------------ #
    def match(self, summary: str) -> List[str]:
        # Nothing to match
        if not self.expert_summaries:
            return []

        if self.mode == RetrievalMode.TFIDF:
            return self.tfidf_matcher.top_k(summary, k=5)  # type: ignore[union-attr]

        if self.mode == RetrievalMode.LLM:
            return self.llm_matcher.select(summary, self.expert_summaries)  # type: ignore[union-attr]

        # HYBRID
        shortlist_ids = self.tfidf_matcher.top_k(summary, k=5)  # type: ignore[union-attr]
        shortlist = [es for es in self.expert_summaries if es.id in shortlist_ids]
        return self.llm_matcher.select(summary, shortlist)  # type: ignore[union-attr]
