"""Offline replay of recorded provider HTTP responses.

``Replay`` is an ``httpx2`` transport handler (the HTTP library the provider
SDKs use) that answers each request with the next fixture from
``fixtures/``. A fixture is ``{"status", "headers", "body"}``. A fixture name
may be replaced by an exception instance to simulate network failures.
Requests are recorded so tests can assert on what was sent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2

from reviewdesk.providers.llm import RetryPolicy

FIXTURES = Path(__file__).parent / "fixtures"

FAKE_KEY = "sk-test-FAKE-KEY-0000000000000000"
"""Placeholder key used offline; never a real credential."""


def load_fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text())
    return data


class Replay:
    def __init__(self, *steps: str | Exception) -> None:
        self.steps = list(steps)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if not self.steps:
            raise AssertionError("Replay ran out of fixtures")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        fx = load_fixture(step)
        return httpx2.Response(
            fx["status"],
            headers=fx.get("headers", {}),
            content=json.dumps(fx["body"]).encode(),
            request=request,
        )

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self))

    def body(self, index: int = -1) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.requests[index].content)
        return data


class Sleeps:
    """Records requested sleeps instead of waiting."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def fast_retry(sleeps: Sleeps, max_attempts: int = 3) -> RetryPolicy:
    return RetryPolicy(max_attempts=max_attempts, base_delay=0.5, sleep=sleeps, rng=lambda: 1.0)
