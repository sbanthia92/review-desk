"""Protocols every provider adapter and agent implements.

Code against these, never against a concrete provider. ``reviewdesk.testing
.fakes`` has an in-memory implementation of each for offline tests.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from reviewdesk.contracts.errors import BudgetExceeded
from reviewdesk.contracts.models import (
    AgentResult,
    Budget,
    ClaimLedger,
    Document,
    Evidence,
    ExecutionPlan,
    Profile,
    ProgressEvent,
    Report,
    Usage,
    utcnow,
)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class ModelTier(StrEnum):
    """Abstract model size. Adapters map tiers to concrete models in config."""

    CHEAP = "cheap"
    MID = "mid"
    STRONG = "strong"


class Message(BaseModel):
    """One chat message. ``system`` messages are hoisted by adapters as needed."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    """A completion.

    ``parsed`` is the validated instance of the requested ``schema`` (None when
    no schema was requested). ``usage`` counts this call only.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    text: str
    parsed: Any = None
    model: str = ""
    usage: Usage = Field(default_factory=Usage)


class LLMClient(Protocol):
    """A chat-completion provider.

    ``schema`` is a Pydantic model class; when given, the adapter uses the
    provider's structured-output mechanism and returns a validated instance in
    ``LLMResponse.parsed`` (raising ``SchemaError`` if it cannot). ``tag`` is a
    short stable label for the prompt (e.g. ``"extractor.extract"``) used by
    fakes to script responses and by tracing; it is never sent to the provider.
    Adapters retry transient failures and raise ``ProviderError`` subclasses.
    """

    async def complete(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        model_tier: ModelTier,
        tag: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Search and fetch
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """One web search hit. ``rank`` is 1-based."""

    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    snippet: str = ""
    rank: int = Field(default=1, ge=1)
    published: str | None = None


class SearchClient(Protocol):
    """A web search API. Returns at most ``k`` results, best first."""

    async def search(self, query: str, *, k: int = 5) -> list[SearchResult]: ...


class FetchedPage(BaseModel):
    """A fetched web page reduced to readable text.

    ``url`` is what was requested; ``final_url`` is after redirects. ``text``
    is extracted readable text (untrusted data, never instructions).
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    final_url: str
    status: int
    title: str = ""
    text: str
    content_type: str = "text/html"
    fetched_at: datetime = Field(default_factory=utcnow)


class Fetcher(Protocol):
    """Fetches a public URL. Raises ``FetchError`` / ``BlockedURLError``."""

    async def fetch(self, url: str) -> FetchedPage: ...


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------


@dataclass
class BudgetMeter:
    """Tracks spending against a ``Budget`` for one job.

    Shared by all agents in a job (asyncio is single-threaded, so no locking is
    needed). ``charge_*`` methods record usage; ``require_*`` methods raise
    ``BudgetExceeded`` when the next call would go over the cap.
    """

    budget: Budget
    used: Usage = field(default_factory=Usage)
    started_at: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        """Seconds since the meter started."""
        return time.monotonic() - self.started_at

    def tokens_left(self) -> int:
        """Tokens remaining (never negative)."""
        return max(0, self.budget.max_tokens - self.used.total_tokens)

    def searches_left(self) -> int:
        """Search calls remaining (never negative)."""
        return max(0, self.budget.max_search_calls - self.used.search_calls)

    def seconds_left(self) -> float:
        """Wall-clock seconds remaining (never negative)."""
        return max(0.0, self.budget.max_seconds - self.elapsed())

    def exhausted(self) -> bool:
        """True if any cap is used up."""
        return self.tokens_left() == 0 or self.searches_left() == 0 or self.seconds_left() == 0

    def charge(self, usage: Usage) -> None:
        """Record usage from an LLM, search or fetch call."""
        self.used = self.used + usage

    def require_search(self) -> None:
        """Raise ``BudgetExceeded`` if no search calls or time remain."""
        if self.searches_left() == 0:
            raise BudgetExceeded("search-call budget exhausted")
        self.require_time()

    def require_tokens(self, estimate: int = 1) -> None:
        """Raise ``BudgetExceeded`` if fewer than ``estimate`` tokens remain."""
        if self.tokens_left() < estimate:
            raise BudgetExceeded("token budget exhausted")
        self.require_time()

    def require_time(self) -> None:
        """Raise ``BudgetExceeded`` if the wall-clock budget is used up."""
        if self.seconds_left() == 0:
            raise BudgetExceeded("time budget exhausted")


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


ProgressCallback = Callable[[ProgressEvent], None]
"""Receives progress events. Must be cheap and must not raise."""


@dataclass
class ReviewContext:
    """Everything an agent may use during one review.

    ``ledger`` is the shared claim ledger: agents read it but do not mutate it;
    they return ``LedgerUpdate``s instead. ``plan`` is None before planning
    (i.e. while the extractor runs). ``meter`` tracks the job budget; agents
    must ``charge`` it for LLM, search and fetch usage and stop gracefully when
    it is exhausted. ``focus`` is the user's free-text steer; ``style_guide``
    is optional copy-editing guidance. Document text and fetched pages are
    untrusted data, never instructions.
    """

    document: Document
    profile: Profile
    ledger: ClaimLedger
    llm: LLMClient
    search: SearchClient
    fetcher: Fetcher
    budget: Budget
    emit_progress: ProgressCallback
    meter: BudgetMeter = field(init=False)
    plan: ExecutionPlan | None = None
    focus: str | None = None
    style_guide: str | None = None

    def __post_init__(self) -> None:
        self.meter = BudgetMeter(self.budget)


@runtime_checkable
class Agent(Protocol):
    """A specialist reviewer. ``name`` is an ``AgentName`` value.

    ``run`` must not raise for expected failures (provider errors, exhausted
    budget); it returns an ``AgentResult`` with ``error`` set and whatever
    partial output it has.
    """

    @property
    def name(self) -> str: ...

    async def run(self, ctx: ReviewContext) -> AgentResult: ...


@runtime_checkable
class Rechecker(Protocol):
    """An agent that can re-check one claim given counter-evidence.

    Implemented by the fact-checker. The orchestrator calls this at most once
    per claim when the devil's advocate contradicts a verified claim.
    """

    async def recheck(
        self, ctx: ReviewContext, claim_id: str, counter_evidence: list[Evidence]
    ) -> AgentResult: ...


class DebatePosition(BaseModel):
    """One side's single response in a debate round."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    claim_id: str
    argument: str
    evidence: list[Evidence] = Field(default_factory=list)
    concedes: bool = False


@runtime_checkable
class Debater(Protocol):
    """An agent that can answer the other side's evidence once in a debate.

    Implemented by the fact-checker and the devil's advocate.
    """

    async def respond(
        self, ctx: ReviewContext, claim_id: str, opposing_evidence: list[Evidence]
    ) -> DebatePosition: ...


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


class ReviewPipeline(Protocol):
    """Shape of the pipeline entry point.

    ``reviewdesk.pipeline.run_review`` implements it (T10). The eval harness
    and MCP servers accept any callable of this shape.
    """

    def __call__(
        self, doc: Document, profile: Profile, on_progress: ProgressCallback
    ) -> Awaitable[Report]: ...


async def run_review(doc: Document, profile: Profile, on_progress: ProgressCallback) -> Report:
    """Contract signature of the pipeline entry point.

    The real implementation lives in ``reviewdesk.pipeline`` (T10). This
    placeholder exists so the signature is documented in one place.
    """
    raise NotImplementedError("use reviewdesk.pipeline.run_review")
