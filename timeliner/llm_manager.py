"""Unified interface for OpenAI or Ollama back-ends."""
from __future__ import annotations

from typing import Any

from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI           # Remote OpenAI models
from langchain_ollama import ChatOllama  # type: ignore

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)


class LLMManager:
    """Factory that hides the differences between LLM providers."""

    def __init__(self, **kwargs: Any):
        settings = get_settings()
        provider = kwargs.get("provider", settings.llm_provider)

        if provider == "openai":
            logger.info("Using OpenAI backend")
            self._client = ChatOpenAI(  # type: ignore
                openai_api_key=settings.openai_api_key,
                temperature=kwargs.get("temperature", 0.2),
                model_name=kwargs.get("model", "gpt-4o-preview"),
            )

        elif provider == "ollama":
            logger.info("Using Ollama backend on http://localhost:11434")
            self._client = ChatOllama(  # type: ignore
                base_url=kwargs.get("base_url", "http://localhost:11434"),
                model=kwargs.get("model", "llama3:latest"),  # e.g. llama3.1
                temperature=kwargs.get("temperature", 0.2),
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ------------------------------------------------------------------
    def chat(self, messages: list[BaseMessage]) -> str:  # noqa: D401
        """Send a list of LangChain messages and return the assistant's reply text."""
        # Use invoke instead of __call__ to avoid deprecation warning
        response = self._client.invoke(messages)  # type: ignore
        return response if isinstance(response, str) else response.content  # type: ignore
