"""The structure reviewer agent.

One mid-tier call over the whole document (structure needs the whole view;
very long documents are reduced to a verbatim outline of headings and
paragraph openings). The model returns issues, each with a verbatim
``anchor``; the agent locates the anchor and returns one
``Severity.STRUCTURE`` finding per issue.

Span policy. Every finding carries a span valid for the document; none uses
``span=None``:

- Issues about existing text (flow, order, assumption) are anchored to the
  quoted passage. If the passage cannot be located the issue is dropped.
- Issues about something missing (missing_section, gap) are anchored to the
  heading of the most relevant section, or the insertion point, as quoted by
  the model. If that quote cannot be located, the issue is still about the
  whole document, so it is anchored to the document's title (first heading,
  else first line) rather than dropped. Both cases are counted in
  ``AgentResult.notes``.

``Profile.DESIGN_DOC`` uses a design-doc prompt (``structure.design_doc``)
that checks rollout, alternatives, risks, failure modes and unstated
assumptions; every other profile uses ``structure.review``. The agent never
mutates the ledger.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from reviewdesk.agents.copyedit.text import TextLocator, title_span
from reviewdesk.agents.structure.prompts import (
    TAG_DESIGN_DOC,
    TAG_REVIEW,
    StructureIssue,
    StructureOutput,
    build_messages,
    kind_label,
)
from reviewdesk.contracts.errors import BudgetExceeded, ProviderError
from reviewdesk.contracts.interfaces import LLMResponse, ModelTier, ReviewContext
from reviewdesk.contracts.models import (
    AgentName,
    AgentResult,
    Finding,
    Profile,
    ProgressEvent,
    ProgressStep,
    Severity,
    Span,
    Usage,
    new_id,
)

log = logging.getLogger(__name__)

DEFAULT_MAX_DOCUMENT_CHARS = 60_000
"""Documents longer than this are sent as an outline instead of in full."""

DEFAULT_MAX_FINDINGS = 15
"""Cap on structure findings kept per document."""

OUTLINE_OPENING_CHARS = 240
"""Longest paragraph opening included in an outline."""

_PROMPT_OVERHEAD_TOKENS = 900
_OUTPUT_ALLOWANCE_TOKENS = 2_000
_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n")
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*(?=\s|$)")


def estimate_tokens(text: str) -> int:
    """Rough token estimate for one structure-review call over ``text``."""
    return len(text) // 4 + _PROMPT_OVERHEAD_TOKENS + _OUTPUT_ALLOWANCE_TOKENS


def build_outline(text: str, opening_chars: int = OUTLINE_OPENING_CHARS) -> str:
    """Outline of ``text``: every heading plus each paragraph's opening.

    Each outline line is copied from ``text`` (whitespace runs collapsed, which
    the locator ignores), so anchors quoted from it can be located in the
    document.
    """
    lines: list[str] = []
    for para in _PARAGRAPH_SPLIT_RE.split(text):
        body: list[str] = []
        for raw in para.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#") and not body:
                lines.append(stripped)
            else:
                body.append(stripped)
        if not body:
            continue
        joined = " ".join(" ".join(body).split())
        m = _SENTENCE_END_RE.search(joined)
        opening = joined[: m.end()] if m else joined
        if len(opening) > opening_chars:
            cut = opening.rfind(" ", 0, opening_chars)
            opening = opening[: cut if cut > 0 else opening_chars]
        lines.append(opening)
    return "\n".join(lines)


@dataclass
class StructureOutcome:
    """Result of ``review``: findings plus counts for notes and tests."""

    findings: list[Finding] = field(default_factory=list)
    unlocated: int = 0
    title_anchored: int = 0
    duplicates: int = 0
    truncated: int = 0
    outline: bool = False
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    def notes(self) -> list[str]:
        """Human-readable notes (counts only, never document text)."""
        notes: list[str] = []
        if self.outline:
            notes.append("structure review used an outline of the long document")
        if self.unlocated:
            notes.append(
                f"{self.unlocated} structure finding(s) quoted text not found in the "
                "document and were dropped"
            )
        if self.title_anchored:
            notes.append(
                f"{self.title_anchored} missing-content finding(s) anchored to the document title"
            )
        if self.duplicates:
            notes.append(f"{self.duplicates} duplicate structure finding(s) merged")
        if self.truncated:
            notes.append(f"{self.truncated} structure finding(s) omitted over the cap")
        return notes


class StructureAgent:
    """Flow, argument-order and missing-section findings.

    Implements the ``Agent`` protocol with ``name == AgentName.STRUCTURE``.
    Uses the design-doc variant for ``Profile.DESIGN_DOC``. Never raises for
    provider or budget failures: returns ``AgentResult.error`` instead.
    """

    def __init__(
        self,
        *,
        max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
        max_findings: int = DEFAULT_MAX_FINDINGS,
        model_tier: ModelTier = ModelTier.MID,
    ) -> None:
        if max_document_chars <= 0 or max_findings <= 0:
            raise ValueError("max_document_chars and max_findings must be positive")
        self.max_document_chars = max_document_chars
        self.max_findings = max_findings
        self.model_tier = model_tier

    @property
    def name(self) -> str:
        """``AgentName.STRUCTURE``."""
        return AgentName.STRUCTURE

    async def run(self, ctx: ReviewContext) -> AgentResult:
        """Review the document's structure and return ``STRUCTURE`` findings."""
        self._emit(ctx, "Reviewing structure", 0.0)
        outcome = await self.review(ctx)
        message = f"Structure review done: {len(outcome.findings)} finding(s)"
        if outcome.error:
            message += " (stopped early)"
        self._emit(ctx, message, 100.0)
        log.info(
            "structure: %d findings, %d unlocated, error=%s",
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

    async def review(self, ctx: ReviewContext) -> StructureOutcome:
        """Run the review call and anchor every issue to a span."""
        doc = ctx.document
        outcome = StructureOutcome()
        fallback = title_span(doc.text)
        if fallback is None:  # nothing to review
            return outcome
        text = doc.text
        if len(text) > self.max_document_chars:
            text = build_outline(doc.text)[: self.max_document_chars]
            outcome.outline = True
        design_doc = ctx.profile is Profile.DESIGN_DOC
        try:
            response = await self._call(ctx, text, design_doc=design_doc, outline=outcome.outline)
        except BudgetExceeded as exc:
            outcome.error = f"budget exhausted: {exc}"
            return outcome
        except ProviderError as exc:
            outcome.error = f"structure review failed: {exc}"
            return outcome
        outcome.usage = response.usage
        parsed = _parse(response)
        if parsed is None:
            outcome.error = "structure review returned malformed output"
            return outcome

        locator = TextLocator(doc.text)
        seen: set[tuple[Span, str, str]] = set()
        for issue in parsed.issues:
            if not issue.message.strip():
                continue
            span = self._anchor(issue, locator, fallback, outcome)
            if span is None:
                continue
            key = (span, issue.kind.value, " ".join(issue.message.lower().split()))
            if key in seen:
                outcome.duplicates += 1
                continue
            seen.add(key)
            outcome.findings.append(self._finding(ctx, issue, span))

        if len(outcome.findings) > self.max_findings:
            outcome.truncated = len(outcome.findings) - self.max_findings
            outcome.findings = outcome.findings[: self.max_findings]
        for f in outcome.findings:  # invariant: every finding has a valid span
            if f.span is None or not f.span.is_valid_for(doc.text):  # pragma: no cover
                raise AssertionError("structure finding without a valid span")
        return outcome

    # -- steps -------------------------------------------------------------

    async def _call(
        self, ctx: ReviewContext, text: str, *, design_doc: bool, outline: bool
    ) -> LLMResponse:
        ctx.meter.require_tokens(estimate_tokens(text))
        response = await ctx.llm.complete(
            build_messages(text, design_doc=design_doc, outline=outline, focus=ctx.focus),
            schema=StructureOutput,
            model_tier=self.model_tier,
            tag=TAG_DESIGN_DOC if design_doc else TAG_REVIEW,
            temperature=0.0,
        )
        ctx.meter.charge(response.usage)
        return response

    def _anchor(
        self,
        issue: StructureIssue,
        locator: TextLocator,
        fallback: Span,
        outcome: StructureOutcome,
    ) -> Span | None:
        located = locator.locate(issue.anchor)
        if located is not None:
            return located.spans[0]
        if issue.kind.is_absence:
            outcome.title_anchored += 1
            return fallback
        outcome.unlocated += 1
        return None

    def _finding(self, ctx: ReviewContext, issue: StructureIssue, span: Span) -> Finding:
        suggestion = (issue.suggestion or "").strip() or None
        return Finding(
            id=new_id("find"),
            agent=self.name,
            severity=Severity.STRUCTURE,
            span=span,
            message=f"{kind_label(issue.kind)}: {issue.message.strip()}",
            suggestion=suggestion,
            claim_ids=[e.claim.id for e in ctx.ledger.claims_overlapping(span)],
        )

    def _emit(self, ctx: ReviewContext, message: str, percent: float) -> None:
        try:
            ctx.emit_progress(
                ProgressEvent(step=ProgressStep.REVIEWING, message=message, percent=percent)
            )
        except Exception:  # progress must never break the review
            log.warning("structure: progress callback raised", exc_info=True)


def _parse(response: LLMResponse) -> StructureOutput | None:
    if isinstance(response.parsed, StructureOutput):
        return response.parsed
    try:
        return StructureOutput.model_validate_json(response.text)
    except ValidationError:
        return None
