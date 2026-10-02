"""Live wiring for the eval harness: the real pipeline and LLM, keys from ``.env``.

Usage::

    uv run python -m eval run -p baseline -p full=eval.live:full \\
        --llm eval.live:make_llm --judge --docs 01-leicester-money

``make_llm`` returns the configured LLM client (for the baseline and the
judge); ``full`` is the real ``run_review`` on agents and providers built by
``reviewdesk.registry.build_registry_from_env``. Variables already set in the
environment win over ``.env``. Keys are never printed.
"""

from __future__ import annotations

import os
from pathlib import Path

from reviewdesk.cli.envfile import load_env_file
from reviewdesk.contracts import Document, LLMClient, Profile, ProgressCallback, Report
from reviewdesk.orchestrator import AgentRegistry
from reviewdesk.pipeline import run_review
from reviewdesk.registry import build_registry_from_env

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """Fill unset (or blank) environment variables from ``path`` if it exists."""
    if not path.is_file():
        return
    for key, value in load_env_file(path).items():
        if value and not os.environ.get(key):
            os.environ[key] = value


def make_llm() -> LLMClient:
    """The configured LLM client, for the baseline reviewer and the judge."""
    load_env()
    return build_registry_from_env().llm


class LivePipeline:
    """``ReviewPipeline`` running the real review; the registry is built once, lazily."""

    def __init__(self) -> None:
        self._registry: AgentRegistry | None = None

    def registry(self) -> AgentRegistry:
        """The real registry (built on first use)."""
        if self._registry is None:
            load_env()
            self._registry = build_registry_from_env()
        return self._registry

    async def __call__(
        self, doc: Document, profile: Profile, on_progress: ProgressCallback
    ) -> Report:
        return await run_review(doc, profile, on_progress, registry=self.registry())


full = LivePipeline()
"""The full pipeline, as ``-p full=eval.live:full``."""
