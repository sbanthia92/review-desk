"""Detect rejected provider keys during a review.

Agents turn provider failures into ``AgentResult.error`` and the orchestrator
turns those into report notes, which would hide an invalid or out-of-credit
key behind an empty "ready" report. The orchestrator wraps the registry's LLM
and search clients in these proxies for each job; they record the first
``AuthError`` (and re-raise it, so agents still degrade as usual), and
``Orchestrator.review`` raises it once the job ends.
"""

from __future__ import annotations

from typing import Any

from reviewdesk.contracts import (
    AuthError,
    LLMClient,
    LLMResponse,
    Message,
    ModelTier,
    SearchClient,
    SearchResult,
)


class AuthWatch:
    """Remembers the first ``AuthError`` raised by a provider in one job."""

    def __init__(self) -> None:
        self.error: AuthError | None = None

    def saw(self, exc: AuthError) -> None:
        """Record ``exc`` unless an earlier error was already recorded."""
        if self.error is None:
            self.error = exc


class WatchedLLM:
    """``LLMClient`` proxy that records ``AuthError`` and re-raises it."""

    def __init__(self, inner: LLMClient, watch: AuthWatch) -> None:
        self._inner = inner
        self._watch = watch

    async def complete(
        self,
        messages: list[Message],
        *,
        schema: Any = None,
        model_tier: ModelTier,
        tag: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Delegate to the wrapped client."""
        try:
            return await self._inner.complete(
                messages,
                schema=schema,
                model_tier=model_tier,
                tag=tag,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except AuthError as exc:
            self._watch.saw(exc)
            raise


class WatchedSearch:
    """``SearchClient`` proxy that records ``AuthError`` and re-raises it."""

    def __init__(self, inner: SearchClient, watch: AuthWatch) -> None:
        self._inner = inner
        self._watch = watch

    async def search(self, query: str, *, k: int = 5) -> list[SearchResult]:
        """Delegate to the wrapped client."""
        try:
            return await self._inner.search(query, k=k)
        except AuthError as exc:
            self._watch.saw(exc)
            raise
