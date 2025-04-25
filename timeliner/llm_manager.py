"""Unified interface for OpenAI or local Llama‑3."""
from __future__ import annotations

from typing import Any, List

from langchain.chat_models import ChatOpenAI  # type: ignore
from langchain.schema import BaseMessage  # type: ignore

try:
    from llama_index.llms import LlamaCPP
except ImportError:  # pragma: no cover
    LlamaCPP = None  # type: ignore

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)


class LLMManager:
    def __init__(self, **kwargs: Any):
        settings = get_settings()
        provider = kwargs.get("provider", settings.llm_provider)

        if provider == "openai":
            logger.info("Using OpenAI backend")
            self._client = ChatOpenAI(
                openai_api_key=settings.openai_api_key,
                temperature=kwargs.get("temperature", 0.2),
                model_name=kwargs.get("model", "gpt-4o-preview"),
            )
        elif provider == "local_llama":
            if LlamaCPP is None:
                raise RuntimeError("llama-index not installed – `pip install llama-index`")
            logger.info("Using local Llama-3 backend")
            self._client = LlamaCPP(
                model_path=kwargs.get("model_path", "./models/llama-3.Q4_K_M.gguf"),
                temperature=kwargs.get("temperature", 0.2),
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ------------------------------------------------------------------
    def chat(self, messages: List[BaseMessage]) -> str:  # noqa: D401
        response = self._client(messages)
        return response if isinstance(response, str) else response.content  # type: ignore[attr-defined]