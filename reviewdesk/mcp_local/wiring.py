"""Pipelines the MCP server can run.

``ReviewRunner`` is the shape the server calls: the contract ``ReviewPipeline``
plus the optional ``focus`` keyword (``reviewdesk.pipeline.run_review``
accepts it). ``adapt_pipeline`` wraps a plain ``ReviewPipeline``.

``EnvPipeline`` is the real wiring. It builds the agent registry lazily, on
the first review, from ``reviewdesk.registry.build_registry_from_env`` (T11).
If that module or function is missing, or raises ``MissingSetup``, the review
fails with a clear missing-setup error instead of crashing the server.

``default_runner`` picks ``EnvPipeline`` or, with ``REVIEWDESK_FAKE=1``, the
offline demo pipeline from ``reviewdesk.mcp_local.demo``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from reviewdesk.contracts import (
    AuthError,
    Document,
    LLMClient,
    LLMResponse,
    Message,
    MissingSetup,
    ModelTier,
    Profile,
    ProgressCallback,
    Report,
    ReviewPipeline,
    SearchClient,
    SearchResult,
)
from reviewdesk.orchestrator import AgentRegistry

log = logging.getLogger(__name__)

REGISTRY_MODULE = "reviewdesk.registry"
REGISTRY_BUILDER = "build_registry_from_env"
FAKE_ENV = "REVIEWDESK_FAKE"
MAX_WORDS_ENV = "REVIEWDESK_MAX_WORDS"


class ReviewRunner(Protocol):
    """``ReviewPipeline`` plus the user's free-text ``focus``."""

    async def __call__(
        self,
        doc: Document,
        profile: Profile,
        on_progress: ProgressCallback,
        *,
        focus: str | None = None,
    ) -> Report: ...


class _PlainPipeline:
    """Adapts a ``ReviewPipeline`` (no ``focus``) to ``ReviewRunner``."""

    def __init__(self, pipeline: ReviewPipeline) -> None:
        self._pipeline = pipeline

    async def __call__(
        self,
        doc: Document,
        profile: Profile,
        on_progress: ProgressCallback,
        *,
        focus: str | None = None,
    ) -> Report:
        return await self._pipeline(doc, profile, on_progress)


def adapt_pipeline(pipeline: ReviewPipeline) -> ReviewRunner:
    """Wrap a contract ``ReviewPipeline``; ``focus`` is dropped."""
    return _PlainPipeline(pipeline)


class RegistryPipeline:
    """Runs ``reviewdesk.pipeline.run_review`` on a registry built on demand.

    ``build`` returns an ``AgentRegistry`` (or an awaitable of one) and may
    raise ``MissingSetup``. A built registry is cached; a failed build is
    retried on the next review (so fixing the env and retrying works once
    the server restarts, and transient failures are not sticky).

    The LLM and search clients are wrapped so that when a provider rejects the
    key (``AuthError``) the review fails with that error instead of returning
    a report degraded to nothing: the orchestrator turns agent failures into
    report notes, which would hide a bad key.
    """

    def __init__(self, build: Callable[[], Any]) -> None:
        self._build = build
        self._registry: AgentRegistry | None = None

    async def _get_registry(self) -> AgentRegistry:
        if self._registry is None:
            built = self._build()
            if inspect.isawaitable(built):
                built = await built
            if not isinstance(built, AgentRegistry):
                raise MissingSetup(
                    f"{REGISTRY_MODULE}.{REGISTRY_BUILDER} did not return an AgentRegistry"
                )
            self._registry = built
        return self._registry

    async def __call__(
        self,
        doc: Document,
        profile: Profile,
        on_progress: ProgressCallback,
        *,
        focus: str | None = None,
    ) -> Report:
        from reviewdesk.pipeline import run_review

        base = await self._get_registry()
        watch = _AuthWatch()
        registry = AgentRegistry(
            llm=_WatchedLLM(base.llm, watch),
            search=_WatchedSearch(base.search, watch),
            fetcher=base.fetcher,
            agents=dict(base.agents),
        )
        report = await run_review(doc, profile, on_progress, registry=registry, focus=focus)
        if watch.error is not None:
            raise watch.error
        return report


def load_registry_builder() -> Callable[[], Any]:
    """Import ``reviewdesk.registry.build_registry_from_env`` lazily.

    Raises ``MissingSetup`` if the module or function does not exist yet.
    """
    try:
        module = importlib.import_module(REGISTRY_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != REGISTRY_MODULE:
            raise
        raise MissingSetup("the real pipeline wiring (reviewdesk.registry) is missing") from exc
    builder: object = getattr(module, REGISTRY_BUILDER, None)
    if not callable(builder):
        raise MissingSetup(f"{REGISTRY_MODULE} has no {REGISTRY_BUILDER}()")
    found: Callable[[], Any] = builder
    return found


class EnvPipeline(RegistryPipeline):
    """The real pipeline, wired from environment variables by T11's builder."""

    def __init__(self) -> None:
        super().__init__(lambda: load_registry_builder()())


def fake_mode(env: Mapping[str, str] | None = None) -> bool:
    """True when ``REVIEWDESK_FAKE`` is set to a truthy value."""
    value = (env if env is not None else os.environ).get(FAKE_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def max_words_from_env(default: int, env: Mapping[str, str] | None = None) -> int:
    """``REVIEWDESK_MAX_WORDS`` as a positive int, else ``default``."""
    raw = (env if env is not None else os.environ).get(MAX_WORDS_ENV, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("ignoring non-integer %s", MAX_WORDS_ENV)
        return default
    return value if value > 0 else default


def default_runner(env: Mapping[str, str] | None = None) -> ReviewRunner:
    """The fake demo pipeline if ``REVIEWDESK_FAKE`` is on, else ``EnvPipeline``."""
    if fake_mode(env):
        from reviewdesk.mcp_local.demo import demo_pipeline

        return demo_pipeline()
    return EnvPipeline()


# ---------------------------------------------------------------------------
# AuthError watch
# ---------------------------------------------------------------------------


class _AuthWatch:
    """Remembers the first ``AuthError`` raised by a provider during a review."""

    def __init__(self) -> None:
        self.error: AuthError | None = None

    def saw(self, exc: AuthError) -> None:
        if self.error is None:
            self.error = exc


class _WatchedLLM:
    """``LLMClient`` proxy that records ``AuthError`` and re-raises it."""

    def __init__(self, inner: LLMClient, watch: _AuthWatch) -> None:
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


class _WatchedSearch:
    """``SearchClient`` proxy that records ``AuthError`` and re-raises it."""

    def __init__(self, inner: SearchClient, watch: _AuthWatch) -> None:
        self._inner = inner
        self._watch = watch

    async def search(self, query: str, *, k: int = 5) -> list[SearchResult]:
        try:
            return await self._inner.search(query, k=k)
        except AuthError as exc:
            self._watch.saw(exc)
            raise
