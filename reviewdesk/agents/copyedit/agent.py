"""The copy editor agent.

Asks a cheap-tier model for line-level edits (verbatim quote plus
replacement), locates each quote in the original text and returns one
``Severity.STYLE`` finding per located edit. Every finding has a span that is
valid for the document, and ``document.text[span.start:span.end]`` is the
quoted passage (matched exactly, or modulo whitespace, quote-mark, dash and
case normalization). Edits that cannot be located, are no-ops, overlap an
earlier edit or are too large to be line edits are dropped and counted in
``AgentResult.notes``. The agent never mutates the ledger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import ValidationError

from reviewdesk.agents.copyedit.prompts import (
    TAG_EDIT,
    CopyEditOutput,
    ProposedEdit,
    build_messages,
)
from reviewdesk.agents.copyedit.text import Chunk, TextLocator, split_into_chunks
from reviewdesk.contracts.errors import BudgetExceeded, ProviderError
from reviewdesk.contracts.interfaces import LLMResponse, ModelTier, ReviewContext
from reviewdesk.contracts.models import (
    AgentName,
    AgentResult,
    Finding,
    ProgressEvent,
    ProgressStep,
    Severity,
    Span,
    Usage,
    new_id,
)

log = logging.getLogger(__name__)

DEFAULT_MAX_CHUNK_CHARS = 8_000
"""Characters of document per copy-edit call (~2k tokens)."""

DEFAULT_MAX_EDITS = 80
"""Cap on edits kept per document (in document order)."""

MAX_QUOTE_CHARS = 400
"""Longer quotes are rewrites, not line edits, and are dropped."""

_PROMPT_OVERHEAD_TOKENS = 700
_OUTPUT_ALLOWANCE_TOKENS = 1_500


def estimate_tokens(text: str, style_guide: str | None = None) -> int:
    """Rough token estimate for one copy-edit call over ``text``."""
    guide = len(style_guide or "")
    return (len(text) + guide) // 4 + _PROMPT_OVERHEAD_TOKENS + _OUTPUT_ALLOWANCE_TOKENS


@dataclass
class EditOutcome:
    """Result of ``edit``: findings plus drop counts for notes and tests."""

    findings: list[Finding] = field(default_factory=list)
    unlocated: int = 0
    no_ops: int = 0
    oversized: int = 0
    overlapping: int = 0
    truncated: int = 0
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    def notes(self) -> list[str]:
        """Human-readable notes (counts only, never document text)."""
        notes: list[str] = []
        if self.unlocated:
            notes.append(
                f"{self.unlocated} copy edit(s) quoted text not found in the document "
                "and were dropped"
            )
        if self.no_ops:
            notes.append(f"{self.no_ops} copy edit(s) that changed nothing were dropped")
        if self.oversized:
            notes.append(f"{self.oversized} copy edit(s) too large for a line edit were dropped")
        if self.overlapping:
            notes.append(f"{self.overlapping} overlapping copy edit(s) were dropped")
        if self.truncated:
            notes.append(f"{self.truncated} copy edit(s) omitted over the cap")
        return notes


class CopyEditAgent:
    """Line-level grammar, clarity and concision suggestions with spans.

    Implements the ``Agent`` protocol with ``name == AgentName.COPYEDIT``.
    Uses ``ctx.style_guide`` when present. Long documents are processed in
    chunks of ``max_chunk_chars``. Never raises for provider or budget
    failures: returns the edits found so far with ``AgentResult.error`` set.
    """

    def __init__(
        self,
        *,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        max_edits: int = DEFAULT_MAX_EDITS,
        model_tier: ModelTier = ModelTier.CHEAP,
    ) -> None:
        if max_chunk_chars <= 0 or max_edits <= 0:
            raise ValueError("max_chunk_chars and max_edits must be positive")
        self.max_chunk_chars = max_chunk_chars
        self.max_edits = max_edits
        self.model_tier = model_tier

    @property
    def name(self) -> str:
        """``AgentName.COPYEDIT``."""
        return AgentName.COPYEDIT

    async def run(self, ctx: ReviewContext) -> AgentResult:
        """Copy-edit the document and return ``STYLE`` findings."""
        self._emit(ctx, "Copy editing", 0.0)
        outcome = await self.edit(ctx)
        message = f"Copy editing done: {len(outcome.findings)} suggestion(s)"
        if outcome.error:
            message += " (stopped early)"
        self._emit(ctx, message, 100.0)
        log.info(
            "copyedit: %d edits, %d unlocated, error=%s",
            len(outcome.findings),
            outcome.unlocated,
            bool(outcome.error),
        )
        return AgentResult(
            agent=self.name,
            findings=outcome.findings,
            usage=outcome.usage,
            error=outcome.error,
            notes=outcome.notes(),
        )

    async def edit(self, ctx: ReviewContext) -> EditOutcome:
        """Run copy editing over every chunk and post-process the edits."""
        doc = ctx.document
        outcome = EditOutcome()
        chunks = [c for c in split_into_chunks(doc.text, self.max_chunk_chars) if c.text.strip()]
        taken: list[Span] = []
        for i, chunk in enumerate(chunks, start=1):
            if len(chunks) > 1:
                self._emit(
                    ctx,
                    f"Copy editing part {i} of {len(chunks)}",
                    100.0 * (i - 1) / len(chunks),
                )
            try:
                response = await self._call(ctx, chunk, i, len(chunks))
            except BudgetExceeded as exc:
                outcome.error = f"budget exhausted: {exc}"
                break
            except ProviderError as exc:
                outcome.error = f"copy edit failed: {exc}"
                break
            outcome.usage = outcome.usage + response.usage
            parsed = _parse(response)
            if parsed is None:
                outcome.error = outcome.error or "copy edit returned malformed output"
                continue
            locator = TextLocator(chunk.text)
            for item in parsed.edits:
                finding = self._to_finding(ctx, item, chunk, locator, taken, outcome)
                if finding is not None:
                    outcome.findings.append(finding)

        outcome.findings.sort(key=lambda f: f.span.start if f.span else 0)
        if len(outcome.findings) > self.max_edits:
            outcome.truncated = len(outcome.findings) - self.max_edits
            outcome.findings = outcome.findings[: self.max_edits]
        for f in outcome.findings:  # invariant: every finding has a valid span
            if f.span is None or not f.span.is_valid_for(doc.text):  # pragma: no cover
                raise AssertionError("copy-edit finding without a valid span")
        return outcome

    # -- steps -------------------------------------------------------------

    async def _call(self, ctx: ReviewContext, chunk: Chunk, part: int, parts: int) -> LLMResponse:
        ctx.meter.require_tokens(estimate_tokens(chunk.text, ctx.style_guide))
        response = await ctx.llm.complete(
            build_messages(
                chunk.text,
                part=part,
                parts=parts,
                style_guide=ctx.style_guide,
                focus=ctx.focus,
            ),
            schema=CopyEditOutput,
            model_tier=self.model_tier,
            tag=TAG_EDIT,
            temperature=0.0,
        )
        ctx.meter.charge(response.usage)
        return response

    def _to_finding(
        self,
        ctx: ReviewContext,
        item: ProposedEdit,
        chunk: Chunk,
        locator: TextLocator,
        taken: list[Span],
        outcome: EditOutcome,
    ) -> Finding | None:
        quote = item.quote.strip()
        if len(quote) > MAX_QUOTE_CHARS or len(item.suggestion) > 2 * MAX_QUOTE_CHARS:
            outcome.oversized += 1
            return None
        # Trimming is not allowed: the suggestion replaces exactly the quote,
        # so a span that silently drops punctuation would corrupt the edit.
        located = locator.locate(quote, allow_trim=False)
        if located is None:
            outcome.unlocated += 1
            return None
        spans = [Span(start=s.start + chunk.start, end=s.end + chunk.start) for s in located.spans]
        # A repeated quote maps to its first occurrence not already edited.
        span = next((s for s in spans if not any(s.overlaps(t) for t in taken)), None)
        if span is None:
            outcome.overlapping += 1
            return None
        original = span.text_of(ctx.document.text)
        suggestion = item.suggestion.strip()
        if suggestion == original.strip():
            outcome.no_ops += 1
            return None
        taken.append(span)
        reason = item.reason.strip() or "Suggested wording change."
        return Finding(
            id=new_id("find"),
            agent=self.name,
            severity=Severity.STYLE,
            span=span,
            message=f"{item.category.capitalize()}: {reason}",
            suggestion=suggestion,
            claim_ids=[e.claim.id for e in ctx.ledger.claims_overlapping(span)],
        )

    def _emit(self, ctx: ReviewContext, message: str, percent: float) -> None:
        try:
            ctx.emit_progress(
                ProgressEvent(step=ProgressStep.REVIEWING, message=message, percent=percent)
            )
        except Exception:  # progress must never break the review
            log.warning("copyedit: progress callback raised", exc_info=True)


def _parse(response: LLMResponse) -> CopyEditOutput | None:
    if isinstance(response.parsed, CopyEditOutput):
        return response.parsed
    try:
        return CopyEditOutput.model_validate_json(response.text)
    except ValidationError:
        return None
