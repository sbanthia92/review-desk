"""Bridge the pipeline's synchronous ``on_progress`` to async MCP progress.

The pipeline calls ``on_progress(event)`` synchronously from inside the event
loop. ``ProgressBridge.callback`` only enqueues the event (never blocks, never
raises); a drain task sends each one, in order, through an async ``send``
function (the tool's ``ctx.report_progress``, which the SDK turns into
``notifications/progress`` only when the client supplied a progress token).
A failing send is logged and skipped: progress never breaks a review.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable
from types import TracebackType

import anyio
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from reviewdesk.contracts import ProgressEvent

log = logging.getLogger(__name__)

PROGRESS_TOTAL = 100.0

SendProgress = Callable[[float, float | None, str | None], Awaitable[None]]
"""``(progress, total, message)``, the shape of ``Context.report_progress``."""


class ProgressBridge:
    """Async context manager that relays ``ProgressEvent``s in order.

    Use ``callback`` as the pipeline's ``on_progress``. On exit the queue is
    drained before returning. Percent values are clamped to 0-100 and kept
    non-decreasing (MCP requires progress to increase).
    """

    def __init__(self, send: SendProgress) -> None:
        self._send = send
        self._last = 0.0
        self._failed = False
        self._tx: MemoryObjectSendStream[ProgressEvent]
        self._rx: MemoryObjectReceiveStream[ProgressEvent]
        self._tx, self._rx = anyio.create_memory_object_stream[ProgressEvent](math.inf)
        self._tg: TaskGroup | None = None

    def callback(self, event: ProgressEvent) -> None:
        """Synchronous ``on_progress``: enqueue ``event``. Never raises."""
        try:
            self._tx.send_nowait(event)
        except Exception:  # closed stream or anything else: drop the event
            log.debug("dropped a progress event")

    def emit(self, step: str, message: str, percent: float) -> None:
        """Enqueue a server-side event (e.g. "Fetching URL")."""
        self.callback(ProgressEvent(step=step, message=message, percent=percent))

    async def _drain(self) -> None:
        async with self._rx:
            async for event in self._rx:
                percent = min(PROGRESS_TOTAL, max(0.0, float(event.percent)))
                self._last = max(self._last, percent)
                try:
                    await self._send(self._last, PROGRESS_TOTAL, event.message)
                except Exception as exc:
                    if not self._failed:
                        log.warning("progress notification failed: %s", type(exc).__name__)
                    self._failed = True

    async def __aenter__(self) -> ProgressBridge:
        self._tg = anyio.create_task_group()
        await self._tg.__aenter__()
        self._tg.start_soon(self._drain)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        self._tx.close()
        assert self._tg is not None
        return await self._tg.__aexit__(exc_type, exc, tb)
