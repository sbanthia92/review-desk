"""Retry policy shared by the search adapters and the fetcher.

The sleep function is injectable so tests run instantly: pass
``RetryPolicy(sleep=no_sleep)`` or any coroutine that records delays.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

Sleep = Callable[[float], Awaitable[None]]


async def no_sleep(_seconds: float) -> None:
    """A sleep that returns immediately (for tests)."""


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff for transient failures.

    ``attempts`` is the total number of tries (1 means no retry). The delay
    before retry ``n`` (0-based) is ``base_delay * 2**n``, capped at
    ``max_delay``. A server-sent ``Retry-After`` replaces the computed delay
    but is still capped, so a hostile or confused server cannot stall a job.
    """

    attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    sleep: Sleep = field(default=asyncio.sleep, compare=False)

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be non-negative")

    def delay(self, retry_index: int, retry_after: float | None = None) -> float:
        """Seconds to wait before retry number ``retry_index`` (0-based)."""
        if retry_after is not None and retry_after >= 0:
            return min(self.max_delay, retry_after)
        return min(self.max_delay, self.base_delay * float(2**retry_index))

    async def wait(self, retry_index: int, retry_after: float | None = None) -> None:
        """Sleep for ``delay(retry_index, retry_after)`` using the injected sleep."""
        await self.sleep(self.delay(retry_index, retry_after))


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header given in seconds; dates are ignored."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None
