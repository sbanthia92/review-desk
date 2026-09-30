"""LLM provider adapters implementing ``reviewdesk.contracts.LLMClient``.

Use ``make_llm_client(provider, api_key)`` or construct ``AnthropicLLM`` /
``OpenAILLM`` directly. Tier -> model defaults live in ``config`` and can be
overridden per client with ``models={...}``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from reviewdesk.contracts import LLMClient, ModelTier
from reviewdesk.providers.llm.anthropic_client import AnthropicLLM
from reviewdesk.providers.llm.config import (
    DEFAULT_ANTHROPIC_MODELS,
    DEFAULT_OPENAI_MODELS,
    RetryPolicy,
)
from reviewdesk.providers.llm.openai_client import OpenAILLM

Provider = Literal["anthropic", "openai"]


def make_llm_client(
    provider: Provider | str,
    api_key: str,
    *,
    models: Mapping[ModelTier | str, str] | None = None,
    **options: Any,
) -> LLMClient:
    """Build the adapter for ``provider`` (``"anthropic"`` or ``"openai"``)."""
    name = provider.lower()
    if name == "anthropic":
        return AnthropicLLM(api_key, models=models, **options)
    if name == "openai":
        return OpenAILLM(api_key, models=models, **options)
    raise ValueError(f"unknown LLM provider: {provider!r}")


__all__ = [
    "DEFAULT_ANTHROPIC_MODELS",
    "DEFAULT_OPENAI_MODELS",
    "AnthropicLLM",
    "OpenAILLM",
    "Provider",
    "RetryPolicy",
    "make_llm_client",
]
