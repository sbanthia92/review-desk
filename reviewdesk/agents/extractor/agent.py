"""The claim extractor agent.

Runs first in every review. Asks a cheap-tier model for verbatim quotes of the
document's thesis, supporting claims and factual claims, locates each quote in
the original text (see ``spans``), and returns one ``AddClaim`` ledger update
per located claim. Quotes that cannot be located are dropped, never given an
invented span. Every emitted claim satisfies
``claim.text == document.text[claim.span.start:claim.span.end]``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from pydantic import ValidationError

from reviewdesk.agents.extractor.chunks import Chunk, split_into_chunks
from reviewdesk.agents.extractor.prompts import (
    TAG_EXTRACT,
    ExtractedClaim,
    ExtractionOutput,
    build_messages,
)
from reviewdesk.agents.extractor.spans import TextLocator
from reviewdesk.contracts.errors import BudgetExceeded, ProviderError
from reviewdesk.contracts.interfaces import LLMResponse, ModelTier, ReviewContext
from reviewdesk.contracts.models import (
    AddClaim,
    AgentName,
    AgentResult,
    Claim,
    ClaimType,
    Document,
    LedgerUpdate,
    ProgressEvent,
    ProgressStep,
    Span,
    Usage,
    new_id,
)

log = logging.getLogger(__name__)

DEFAULT_MAX_CHUNK_CHARS = 12_000
"""Characters of document per extraction call (~3k tokens)."""

DEFAULT_MAX_CLAIMS = 60
"""Cap on claims kept per document (most important first)."""

MIN_CLAIM_CHARS = 3
"""Located spans shorter than this (after stripping) are discarded as noise."""

MAX_STANDALONE_CHARS = 400
"""Cap on a claim's standalone restatement."""

_PROMPT_OVERHEAD_TOKENS = 600
_OUTPUT_ALLOWANCE_TOKENS = 1_500
_TYPE_RANK = {ClaimType.THESIS: 0, ClaimType.SUPPORTING: 1, ClaimType.FACTUAL: 2}


def clamp_importance(value: float) -> float:
    """Clamp to [0, 1]; non-finite values become 0.5 (neutral)."""
    if not math.isfinite(value):
        return 0.5
    return min(1.0, max(0.0, value))


def clean_standalone(value: str) -> str:
    """One line, clipped: the model's standalone restatement of a claim."""
    return " ".join(value.split())[:MAX_STANDALONE_CHARS]


def estimate_tokens(text: str) -> int:
    """Rough token estimate for one extraction call over ``text``."""
    return len(text) // 4 + _PROMPT_OVERHEAD_TOKENS + _OUTPUT_ALLOWANCE_TOKENS


@dataclass
class _Candidate:
    span: Span
    type: ClaimType
    importance: float
    order: int
    standalone: str = ""


@dataclass
class ExtractionOutcome:
    """Result of ``extract_claims``: claims plus counts for notes and tests."""

    claims: list[Claim] = field(default_factory=list)
    unlocated: int = 0
    duplicates: int = 0
    demoted_theses: int = 0
    discarded: int = 0
    truncated: int = 0
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    def notes(self) -> list[str]:
        """Human-readable notes (counts only, never document text)."""
        notes: list[str] = []
        if self.unlocated:
            notes.append(
                f"{self.unlocated} quoted claim(s) could not be located in the "
                "document and were dropped"
            )
        if self.duplicates:
            notes.append(f"{self.duplicates} duplicate claim(s) merged")
        if self.demoted_theses:
            notes.append(f"{self.demoted_theses} extra thesis claim(s) treated as supporting")
        if self.discarded:
            notes.append(f"{self.discarded} fragment(s) too short to be claims discarded")
        if self.truncated:
            notes.append(f"{self.truncated} low-importance claim(s) omitted over the cap")
        return notes


class ExtractorAgent:
    """Splits a document into thesis, supporting and factual claims.

    Implements the ``Agent`` protocol with ``name == AgentName.EXTRACTOR``.
    Long documents are processed in chunks of ``max_chunk_chars``. At most
    one thesis is kept (the most important); at most ``max_claims`` claims.
    Never raises for provider or budget failures: returns partial claims with
    ``AgentResult.error`` set.
    """

    def __init__(
        self,
        *,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        max_claims: int = DEFAULT_MAX_CLAIMS,
        model_tier: ModelTier = ModelTier.CHEAP,
    ) -> None:
        if max_chunk_chars <= 0 or max_claims <= 0:
            raise ValueError("max_chunk_chars and max_claims must be positive")
        self.max_chunk_chars = max_chunk_chars
        self.max_claims = max_claims
        self.model_tier = model_tier

    @property
    def name(self) -> str:
        """``AgentName.EXTRACTOR``."""
        return AgentName.EXTRACTOR

    async def run(self, ctx: ReviewContext) -> AgentResult:
        """Extract claims and return them as ``AddClaim`` updates."""
        self._emit(ctx, "Extracting claims", 0.0)
        outcome = await self.extract_claims(ctx)
        message = f"Extracted {len(outcome.claims)} claim(s)"
        notes = outcome.notes()
        if notes:
            message += "; " + "; ".join(notes)
        if outcome.error:
            message += f" (stopped early: {outcome.error})"
        self._emit(ctx, message, 100.0)
        log.info(
            "extractor: %d claims, %d unlocated, %d duplicates, error=%s",
            len(outcome.claims),
            outcome.unlocated,
            outcome.duplicates,
            bool(outcome.error),
        )
        updates: list[LedgerUpdate] = [AddClaim(claim=c) for c in outcome.claims]
        return AgentResult(
            agent=self.name, ledger_updates=updates, usage=outcome.usage, error=outcome.error
        )

    async def extract_claims(self, ctx: ReviewContext) -> ExtractionOutcome:
        """Run extraction over every chunk and post-process the claims."""
        doc = ctx.document
        outcome = ExtractionOutcome()
        chunks = [c for c in split_into_chunks(doc.text, self.max_chunk_chars) if c.text.strip()]
        candidates: list[_Candidate] = []
        used: set[Span] = set()
        for i, chunk in enumerate(chunks, start=1):
            if len(chunks) > 1:
                self._emit(
                    ctx,
                    f"Extracting claims from part {i} of {len(chunks)}",
                    100.0 * (i - 1) / len(chunks),
                )
            try:
                response = await self._call(ctx, chunk, i, len(chunks))
            except BudgetExceeded as exc:
                outcome.error = f"budget exhausted: {exc}"
                break
            except ProviderError as exc:
                # Adapters already retried transient failures; stop here and
                # keep whatever earlier chunks produced.
                outcome.error = f"claim extraction failed: {exc}"
                break
            outcome.usage = outcome.usage + response.usage
            parsed = _parse(response)
            if parsed is None:
                outcome.error = outcome.error or "claim extraction returned malformed output"
                continue
            locator = TextLocator(chunk.text)
            for item in parsed.claims:
                cand = self._locate(item, chunk, locator, used, len(candidates))
                if cand is None:
                    outcome.unlocated += 1
                    continue
                if cand.span in used:
                    outcome.duplicates += 1
                used.add(cand.span)
                candidates.append(cand)
        outcome.claims = self._finalize(doc, candidates, outcome)
        return outcome

    # -- steps -------------------------------------------------------------

    async def _call(self, ctx: ReviewContext, chunk: Chunk, part: int, parts: int) -> LLMResponse:
        ctx.meter.require_tokens(estimate_tokens(chunk.text))
        response = await ctx.llm.complete(
            build_messages(chunk.text, part=part, parts=parts, focus=ctx.focus),
            schema=ExtractionOutput,
            model_tier=self.model_tier,
            tag=TAG_EXTRACT,
            temperature=0.0,
        )
        ctx.meter.charge(response.usage)
        return response

    def _locate(
        self,
        item: ExtractedClaim,
        chunk: Chunk,
        locator: TextLocator,
        used: set[Span],
        order: int,
    ) -> _Candidate | None:
        located = locator.locate(item.quote)
        if located is None:
            return None
        spans = [Span(start=s.start + chunk.start, end=s.end + chunk.start) for s in located.spans]
        # Repeated quotes map to successive occurrences; if all are taken the
        # claim is a duplicate of the first.
        span = next((s for s in spans if s not in used), spans[0])
        return _Candidate(
            span=span,
            type=item.type,
            importance=clamp_importance(item.importance),
            order=order,
            standalone=clean_standalone(item.standalone),
        )

    def _finalize(
        self, doc: Document, candidates: list[_Candidate], outcome: ExtractionOutcome
    ) -> list[Claim]:
        # Merge duplicates by span: keep the highest importance, then the
        # more central type, then the first mention.
        best: dict[Span, _Candidate] = {}
        for cand in candidates:
            current = best.get(cand.span)
            if current is None or _better(cand, current):
                best[cand.span] = cand
        merged = list(best.values())

        # At most one thesis: keep the most important, demote the rest.
        theses = sorted(
            (c for c in merged if c.type is ClaimType.THESIS),
            key=lambda c: (-c.importance, c.span.start),
        )
        for extra in theses[1:]:
            extra.type = ClaimType.SUPPORTING
            outcome.demoted_theses += 1

        # Drop noise and anything that fails the round-trip check.
        kept: list[_Candidate] = []
        for cand in merged:
            text = doc.text[cand.span.start : cand.span.end]
            if len(text.strip()) < MIN_CLAIM_CHARS or text != text.strip():
                outcome.discarded += 1
                continue
            kept.append(cand)

        # Cap the count, always keeping the thesis.
        kept.sort(key=lambda c: (_TYPE_RANK[c.type] != 0, -c.importance, c.span.start))
        if len(kept) > self.max_claims:
            outcome.truncated = len(kept) - self.max_claims
            kept = kept[: self.max_claims]

        kept.sort(key=lambda c: (c.span.start, c.span.end))
        claims = [
            Claim(
                id=new_id("claim"),
                text=doc.text[c.span.start : c.span.end],
                span=c.span,
                type=c.type,
                importance=c.importance,
                standalone=""
                if c.standalone == doc.text[c.span.start : c.span.end]
                else c.standalone,
            )
            for c in kept
        ]
        for claim in claims:
            if claim.span.text_of(doc.text) != claim.text:  # pragma: no cover - invariant
                raise AssertionError("claim span does not round-trip")
        return claims

    def _emit(self, ctx: ReviewContext, message: str, percent: float) -> None:
        try:
            ctx.emit_progress(
                ProgressEvent(step=ProgressStep.EXTRACTING, message=message, percent=percent)
            )
        except Exception:  # progress must never break extraction
            log.warning("extractor: progress callback raised", exc_info=True)


def _better(a: _Candidate, b: _Candidate) -> bool:
    """True if duplicate ``a`` should replace ``b``."""
    return (-a.importance, _TYPE_RANK[a.type], a.order) < (
        -b.importance,
        _TYPE_RANK[b.type],
        b.order,
    )


def _parse(response: LLMResponse) -> ExtractionOutput | None:
    if isinstance(response.parsed, ExtractionOutput):
        return response.parsed
    try:
        return ExtractionOutput.model_validate_json(response.text)
    except ValidationError:
        return None
