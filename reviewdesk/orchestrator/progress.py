"""Overall progress: one monotonic 0-100 sequence across the job's steps.

Each ``ProgressStep`` owns a slice of the 0-100 range. Agents emit their own
0-100 events; ``ProgressTracker`` rescales them into the current step's slice
(averaging over agents running in parallel) and never lets the overall percent
go backwards. Exceptions from the caller's callback are swallowed.
"""

from __future__ import annotations

import logging

from reviewdesk.contracts import ProgressCallback, ProgressEvent, ProgressStep

log = logging.getLogger(__name__)

STEP_ORDER: tuple[ProgressStep, ...] = (
    ProgressStep.CLASSIFYING,
    ProgressStep.EXTRACTING,
    ProgressStep.PLANNING,
    ProgressStep.REVIEWING,
    ProgressStep.REACTING,
    ProgressStep.RESOLVING,
    ProgressStep.REPORTING,
    ProgressStep.DONE,
)
"""The order steps are emitted in; every job emits each of them."""

STEP_RANGES: dict[ProgressStep, tuple[float, float]] = {
    ProgressStep.CLASSIFYING: (0.0, 4.0),
    ProgressStep.EXTRACTING: (4.0, 25.0),
    ProgressStep.PLANNING: (25.0, 30.0),
    ProgressStep.REVIEWING: (30.0, 80.0),
    ProgressStep.REACTING: (80.0, 90.0),
    ProgressStep.RESOLVING: (90.0, 95.0),
    ProgressStep.REPORTING: (95.0, 99.0),
    ProgressStep.DONE: (100.0, 100.0),
}
"""The overall percent range each step maps its local 0-1 progress into."""


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


class ProgressTracker:
    """Maps step-local progress onto one monotonic overall percent."""

    def __init__(self, callback: ProgressCallback | None) -> None:
        self._callback = callback
        self._last = 0.0
        self._step_index = -1
        self._parts: dict[str, float] = {}
        self.events: list[ProgressEvent] = []

    @property
    def percent(self) -> float:
        """The last overall percent emitted."""
        return self._last

    def emit(self, step: ProgressStep, message: str, fraction: float = 0.0) -> None:
        """Emit ``message`` at ``fraction`` (0-1) of ``step``'s range.

        Steps only move forward: an event for an earlier step is dropped.
        """
        index = STEP_ORDER.index(step)
        if index < self._step_index:
            return
        if index > self._step_index:
            self._step_index = index
            self._parts = {}
        lo, hi = STEP_RANGES[step]
        percent = round(lo + (hi - lo) * _clamp(fraction), 1)
        percent = max(percent, self._last)
        self._last = percent
        event = ProgressEvent(step=step.value, message=message, percent=percent)
        self.events.append(event)
        if self._callback is None:
            return
        try:
            self._callback(event)
        except Exception:  # a broken progress sink must never fail the job
            log.warning("orchestrator: progress callback raised", exc_info=True)

    def start_parts(self, step: ProgressStep, keys: list[str], message: str) -> None:
        """Enter ``step`` with parallel parts ``keys`` (each 0 done)."""
        self.emit(step, message, 0.0)
        self._parts = dict.fromkeys(keys, 0.0)

    def _fraction(self) -> float:
        if not self._parts:
            return 0.0
        return sum(self._parts.values()) / len(self._parts)

    def part(self, step: ProgressStep, key: str, fraction: float, message: str) -> None:
        """Record part ``key`` of ``step`` at ``fraction`` and emit overall progress.

        Ignored once the job has moved past ``step`` (late agent events).
        """
        if STEP_ORDER.index(step) != self._step_index:
            return
        self._parts[key] = max(self._parts.get(key, 0.0), _clamp(fraction))
        self.emit(step, message, self._fraction())

    def callback_for(self, step: ProgressStep, key: str) -> ProgressCallback:
        """A ``ProgressCallback`` for one agent: rescales its 0-100 events."""

        def forward(event: ProgressEvent) -> None:
            try:
                self.part(step, key, event.percent / 100.0, event.message)
            except Exception:  # agents must never be broken by progress
                log.warning("orchestrator: progress forwarding failed", exc_info=True)

        return forward
