"""In-memory fakes for every contract interface. Offline, deterministic.

Typical use::

    llm = FakeLLM({"extractor.extract": {"claims": [...]}})
    ctx = make_context(OPINION_DOC, llm=llm)
    result = await ExtractorAgent().run(ctx)
    assert llm.calls[0].tag == "extractor.extract"
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from reviewdesk.contracts.errors import FetchError, SchemaError
from reviewdesk.contracts.interfaces import (
    DebatePosition,
    FetchedPage,
    LLMResponse,
    Message,
    ModelTier,
    ProgressCallback,
    ReviewContext,
    SearchResult,
)
from reviewdesk.contracts.models import (
    AgentResult,
    Budget,
    ClaimLedger,
    Document,
    Evidence,
    ExecutionPlan,
    Finding,
    LedgerUpdate,
    Profile,
    ProgressEvent,
    Usage,
)
from reviewdesk.testing.samples import (
    DESIGN_DOC,
    ESSAY_DOC,
    FIXED_TIME,
    OPINION_DOC,
    REPORT_DOC,
    SHORT_DOC,
    claim_for,
    empty_report,
    sample_documents,
    sample_ledger,
    sample_report,
    span_of,
)

__all__ = [
    "DESIGN_DOC",
    "ESSAY_DOC",
    "FIXED_TIME",
    "OPINION_DOC",
    "REPORT_DOC",
    "SHORT_DOC",
    "FakeAgent",
    "FakeDevilsAdvocate",
    "FakeFactChecker",
    "FakeFetcher",
    "FakeLLM",
    "FakeLLMCall",
    "FakeLLMUnscripted",
    "FakeSearch",
    "ProgressRecorder",
    "Scripted",
    "claim_for",
    "empty_report",
    "make_context",
    "sample_documents",
    "sample_ledger",
    "sample_report",
    "span_of",
]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

Scripted = str | dict[str, Any] | list[Any] | BaseModel | Exception | Callable[..., Any]
"""A scripted LLM reply.

- ``str``: returned as text (parsed as JSON when a schema is requested).
- ``dict`` / ``list`` / ``BaseModel``: validated against the requested schema;
  the text is its JSON.
- ``Exception``: raised from ``complete``.
- callable: called with ``(messages, schema)`` and its return is handled as
  above. Use it for replies that depend on the prompt.
"""


class FakeLLMUnscripted(AssertionError):
    """Raised when ``FakeLLM`` gets a tag with no scripted reply."""


@dataclass
class FakeLLMCall:
    """One recorded ``FakeLLM.complete`` call."""

    messages: list[Message]
    schema: type[BaseModel] | None
    model_tier: ModelTier
    tag: str


class FakeLLM:
    """Scripted ``LLMClient``: replies are keyed by prompt tag.

    Lookup order: exact tag, then the longest dotted prefix (``"factcheck"``
    matches ``"factcheck.judge"``), then ``default``. A list value is a queue
    of ``Scripted`` replies, consumed in order; the last one repeats. To script
    a reply that *is* a JSON list, wrap it: ``[[...]]``.

    Every call is recorded in ``calls``. Usage counts whitespace-split words
    as tokens and one ``llm_calls`` per call.
    """

    def __init__(
        self,
        script: dict[str, Scripted | Sequence[Scripted]] | None = None,
        *,
        default: Scripted | None = None,
    ) -> None:
        self.script: dict[str, Any] = dict(script or {})
        self.default = default
        self.calls: list[FakeLLMCall] = []
        self._cursor: dict[str, int] = {}

    def calls_for(self, tag: str) -> list[FakeLLMCall]:
        """Recorded calls with exactly this tag."""
        return [c for c in self.calls if c.tag == tag]

    def _lookup(self, tag: str) -> tuple[str, Any]:
        if tag in self.script:
            return tag, self.script[tag]
        parts = tag.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in self.script:
                return prefix, self.script[prefix]
        if self.default is not None:
            return "", self.default
        raise FakeLLMUnscripted(f"FakeLLM has no scripted reply for tag {tag!r}")

    def _next(self, tag: str) -> Any:
        key, value = self._lookup(tag)
        if isinstance(value, list):
            if not value:
                raise FakeLLMUnscripted(f"FakeLLM script for {key!r} is empty")
            i = self._cursor.get(key, 0)
            self._cursor[key] = i + 1
            return value[min(i, len(value) - 1)]
        return value

    async def complete(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        model_tier: ModelTier,
        tag: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Return the next scripted reply for ``tag``."""
        self.calls.append(FakeLLMCall(list(messages), schema, model_tier, tag))
        reply = self._next(tag)
        if callable(reply) and not isinstance(reply, BaseModel | type):
            reply = reply(messages, schema)
        if isinstance(reply, Exception):
            raise reply
        text, parsed = _render(reply, schema)
        usage = Usage(
            input_tokens=sum(len(m.content.split()) for m in messages),
            output_tokens=len(text.split()),
            llm_calls=1,
        )
        return LLMResponse(text=text, parsed=parsed, model=f"fake-{model_tier}", usage=usage)


def _render(reply: Any, schema: type[BaseModel] | None) -> tuple[str, Any]:
    if isinstance(reply, BaseModel):
        text = reply.model_dump_json()
        data: Any = reply.model_dump(mode="json")
    elif isinstance(reply, str):
        text = reply
        if schema is None:
            return text, None
        try:
            data = json.loads(reply)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"scripted reply is not JSON: {exc}") from exc
    else:
        data = reply
        text = json.dumps(reply, default=str)
    if schema is None:
        return text, None
    try:
        return text, schema.model_validate(data)
    except ValueError as exc:
        raise SchemaError(f"scripted reply does not match {schema.__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# Search and fetch
# ---------------------------------------------------------------------------


class FakeSearch:
    """Scripted ``SearchClient``.

    ``results`` maps a query key to hits. A query matches its exact key first,
    then any key contained in it (case-insensitive, longest key wins), then
    ``default``. ``error`` makes every call raise. Queries are recorded in
    ``queries``.
    """

    def __init__(
        self,
        results: dict[str, list[SearchResult]] | None = None,
        *,
        default: list[SearchResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = dict(results or {})
        self.default = list(default or [])
        self.error = error
        self.queries: list[str] = []

    async def search(self, query: str, *, k: int = 5) -> list[SearchResult]:
        """Return scripted hits for ``query`` (at most ``k``)."""
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        if query in self.results:
            return self.results[query][:k]
        lowered = query.lower()
        matches = [key for key in self.results if key.lower() in lowered]
        if matches:
            return self.results[max(matches, key=len)][:k]
        return self.default[:k]


class FakeFetcher:
    """Scripted ``Fetcher``: ``pages`` maps URL to a page or to plain text.

    Unknown URLs raise ``FetchError``; a value that is an ``Exception`` is
    raised. Requested URLs are recorded in ``fetched``.
    """

    def __init__(self, pages: dict[str, FetchedPage | str | Exception] | None = None) -> None:
        self.pages = dict(pages or {})
        self.fetched: list[str] = []

    async def fetch(self, url: str) -> FetchedPage:
        """Return the scripted page for ``url``."""
        self.fetched.append(url)
        page = self.pages.get(url)
        if page is None:
            raise FetchError(f"unreachable: {url}")
        if isinstance(page, Exception):
            raise page
        if isinstance(page, str):
            return FetchedPage(
                url=url, final_url=url, status=200, title=url, text=page, fetched_at=FIXED_TIME
            )
        return page


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class FakeAgent:
    """Configurable ``Agent`` returning canned output.

    ``delay`` sleeps before returning (to test parallelism and timeouts).
    ``raises`` makes ``run`` raise (to test degradation). ``progress``
    messages are emitted through ``ctx.emit_progress``. ``on_run`` may compute
    the result from the context instead. ``runs`` records each context.
    """

    def __init__(
        self,
        name: str,
        *,
        findings: list[Finding] | None = None,
        ledger_updates: list[LedgerUpdate] | None = None,
        usage: Usage | None = None,
        delay: float = 0.0,
        raises: Exception | None = None,
        error: str | None = None,
        progress: list[str] | None = None,
        on_run: Callable[[ReviewContext], AgentResult] | None = None,
    ) -> None:
        self._name = name
        self.findings = list(findings or [])
        self.ledger_updates = list(ledger_updates or [])
        self.usage = usage or Usage()
        self.delay = delay
        self.raises = raises
        self.error = error
        self.progress = list(progress or [])
        self.on_run = on_run
        self.runs: list[ReviewContext] = []

    @property
    def name(self) -> str:
        """The agent's name."""
        return self._name

    async def run(self, ctx: ReviewContext) -> AgentResult:
        """Return the canned result."""
        self.runs.append(ctx)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        for message in self.progress:
            ctx.emit_progress(ProgressEvent(step=self._name, message=message, percent=50.0))
        if self.on_run is not None:
            return self.on_run(ctx)
        return AgentResult(
            agent=self._name,
            findings=list(self.findings),
            ledger_updates=list(self.ledger_updates),
            usage=self.usage,
            error=self.error,
        )


class FakeFactChecker(FakeAgent):
    """``FakeAgent`` that is also a ``Rechecker`` and a ``Debater``.

    ``recheck_result`` is returned by ``recheck``; ``debate_position`` by
    ``respond``. Calls are recorded in ``rechecks`` and ``debates``.
    """

    def __init__(
        self,
        name: str = "factcheck",
        *,
        recheck_result: AgentResult | None = None,
        debate_position: DebatePosition | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.recheck_result = recheck_result
        self.debate_position = debate_position
        self.rechecks: list[tuple[str, list[Evidence]]] = []
        self.debates: list[tuple[str, list[Evidence]]] = []

    async def recheck(
        self, ctx: ReviewContext, claim_id: str, counter_evidence: list[Evidence]
    ) -> AgentResult:
        """Record the call and return ``recheck_result`` (empty by default)."""
        self.rechecks.append((claim_id, list(counter_evidence)))
        return self.recheck_result or AgentResult(agent=self.name)

    async def respond(
        self, ctx: ReviewContext, claim_id: str, opposing_evidence: list[Evidence]
    ) -> DebatePosition:
        """Record the call and return ``debate_position`` (holds by default)."""
        self.debates.append((claim_id, list(opposing_evidence)))
        return self.debate_position or DebatePosition(
            agent=self.name, claim_id=claim_id, argument="The original verdict stands."
        )


class FakeDevilsAdvocate(FakeAgent):
    """``FakeAgent`` that is also a ``Debater``."""

    def __init__(
        self,
        name: str = "devils_advocate",
        *,
        debate_position: DebatePosition | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.debate_position = debate_position
        self.debates: list[tuple[str, list[Evidence]]] = []

    async def respond(
        self, ctx: ReviewContext, claim_id: str, opposing_evidence: list[Evidence]
    ) -> DebatePosition:
        """Record the call and return ``debate_position`` (holds by default)."""
        self.debates.append((claim_id, list(opposing_evidence)))
        return self.debate_position or DebatePosition(
            agent=self.name, claim_id=claim_id, argument="The rebuttal stands."
        )


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


@dataclass
class ProgressRecorder:
    """A ``ProgressCallback`` that stores every event in ``events``."""

    events: list[ProgressEvent] = field(default_factory=list)

    def __call__(self, event: ProgressEvent) -> None:
        self.events.append(event)

    @property
    def steps(self) -> list[str]:
        """The ``step`` of each event, in order."""
        return [e.step for e in self.events]


def make_context(
    document: Document | None = None,
    *,
    profile: Profile = Profile.OPINION,
    ledger: ClaimLedger | None = None,
    llm: FakeLLM | None = None,
    search: FakeSearch | None = None,
    fetcher: FakeFetcher | None = None,
    budget: Budget | None = None,
    emit_progress: ProgressCallback | None = None,
    plan: ExecutionPlan | None = None,
    focus: str | None = None,
    style_guide: str | None = None,
) -> ReviewContext:
    """Build a ``ReviewContext`` with fakes for anything not supplied.

    Defaults: ``OPINION_DOC``, an empty ledger, an unscripted ``FakeLLM``,
    empty search and fetch fakes, the default ``Budget`` and a
    ``ProgressRecorder``.
    """
    return ReviewContext(
        document=document or OPINION_DOC,
        profile=profile,
        ledger=ledger if ledger is not None else ClaimLedger(),
        llm=llm or FakeLLM(),
        search=search or FakeSearch(),
        fetcher=fetcher or FakeFetcher(),
        budget=budget or Budget(),
        emit_progress=emit_progress or ProgressRecorder(),
        plan=plan,
        focus=focus,
        style_guide=style_guide,
    )
