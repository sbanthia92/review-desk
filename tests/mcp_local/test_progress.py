"""``ProgressBridge``: sync callback to ordered async sends, failure-tolerant."""

from __future__ import annotations

import anyio.lowlevel

from reviewdesk.contracts import ProgressEvent
from reviewdesk.mcp_local.progress import ProgressBridge


def event(message: str, percent: float) -> ProgressEvent:
    return ProgressEvent(step="reviewing", message=message, percent=percent)


async def test_events_are_sent_in_order_and_drained_on_exit() -> None:
    sent: list[tuple[float, float | None, str | None]] = []

    async def send(progress: float, total: float | None, message: str | None) -> None:
        await anyio.lowlevel.checkpoint()
        sent.append((progress, total, message))

    async with ProgressBridge(send) as bridge:
        for i in range(20):
            bridge.callback(event(f"step {i}", i * 5.0))

    assert sent == [(i * 5.0, 100.0, f"step {i}") for i in range(20)]


async def test_a_failing_send_does_not_break_anything() -> None:
    sent: list[str | None] = []

    async def send(progress: float, total: float | None, message: str | None) -> None:
        if message == "bad":
            raise ConnectionError("client went away")
        sent.append(message)

    async with ProgressBridge(send) as bridge:
        bridge.callback(event("one", 10))
        bridge.callback(event("bad", 20))
        bridge.callback(event("three", 30))

    assert sent == ["one", "three"]


async def test_callback_after_exit_is_ignored() -> None:
    async def send(progress: float, total: float | None, message: str | None) -> None:
        return None

    async with ProgressBridge(send) as bridge:
        pass
    bridge.callback(event("late", 50))  # must not raise
    bridge.emit("reviewing", "late too", 60)
