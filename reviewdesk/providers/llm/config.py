"""Configuration for LLM adapters: tier -> model mapping and retry policy.

Every adapter takes an optional ``models`` mapping that overrides the defaults
per tier, so callers can pin models without code changes.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from reviewdesk.contracts import ModelTier

DEFAULT_ANTHROPIC_MODELS: Mapping[ModelTier, str] = {
    ModelTier.CHEAP: "claude-haiku-4-5-20251001",
    ModelTier.MID: "claude-sonnet-5-5",
    ModelTier.STRONG: "claude-opus-5-5",
}

DEFAULT_OPENAI_MODELS: Mapping[ModelTier, str] = {
    ModelTier.CHEAP: "gpt-5.4-mini",
    ModelTier.MID: "gpt-5.4",
    ModelTier.STRONG: "gpt-5.5",
}

DEFAULT_ANTHROPIC_EFFORT: Mapping[ModelTier, str] = {
    ModelTier.MID: "medium",
    ModelTier.STRONG: "medium",
}
"""``output_config.effort`` per tier (thinking depth and token spend).

Only sent to models that accept it (see ``anthropic_supports_effort``).
"""

ANTHROPIC_THINKING_HEADROOM = 8_000
"""Extra ``max_tokens`` for models whose thinking is always on.

Thinking tokens count against ``max_tokens``, so a caller asking for a
300-token answer would otherwise be cut off before any output.
"""

_NO_EFFORT_PREFIXES = ("claude-haiku", "claude-3", "claude-sonnet-4-5")


def anthropic_supports_effort(model: str) -> bool:
    """True for models that accept ``output_config.effort`` and think by default."""
    return not model.startswith(_NO_EFFORT_PREFIXES)


DEFAULT_MAX_TOKENS = 4096
"""Output-token cap used when the caller does not pass ``max_tokens``."""

DEFAULT_TIMEOUT_SECONDS = 120.0
"""Per-request timeout handed to the provider SDK."""


def resolve_models(
    defaults: Mapping[ModelTier, str],
    overrides: Mapping[ModelTier | str, str] | None,
) -> dict[ModelTier, str]:
    """Merge per-tier overrides (keyed by ``ModelTier`` or its string value)."""
    merged = dict(defaults)
    for tier, model in (overrides or {}).items():
        merged[ModelTier(tier)] = model
    return merged


Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter for transient provider failures.

    ``max_attempts`` counts the first try. ``sleep`` and ``rng`` are injectable
    so tests never really wait. A provider ``retry-after`` hint is honoured but
    capped at ``max_delay``.
    """

    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0
    sleep: Sleep = asyncio.sleep
    rng: Callable[[], float] = field(default=random.random)

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait after failed attempt number ``attempt`` (0-based)."""
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self.max_delay)
        ceiling = min(self.max_delay, self.base_delay * (2**attempt))
        return float(ceiling * (0.5 + 0.5 * self.rng()))
