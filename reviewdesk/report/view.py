"""Format-neutral view model built from a ``Report``.

Every renderer (markdown, HTML, plain text) consumes the same ``ReportView``,
so section order, grouping, quoting and link safety are decided once, here.

All strings in the view are *untrusted* (document text, claim text, agent
messages, evidence titles and excerpts). This module only normalizes them
(control characters stripped, quotes collapsed to one line and truncated);
each renderer is responsible for escaping for its own output format.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC
from urllib.parse import quote, urlsplit

from reviewdesk.contracts.models import (
    Document,
    Evidence,
    Finding,
    LedgerEntry,
    Rebuttal,
    Report,
    Severity,
    Span,
    Verdict,
)

MAX_QUOTE_CHARS = 300
"""Quoted spans and claim texts longer than this are truncated with an ellipsis."""

MAX_EXCERPT_CHARS = 300
"""Evidence excerpts longer than this are truncated with an ellipsis."""

MAX_COUNTER_CASE = 3
"""The counter-case section shows at most this many rebuttals (design doc)."""

EMPTY_SECTION = "None found."
"""Line rendered for every empty section, in every format."""

ORIGINALITY_DISCLAIMER = (
    "Heuristic: these are automated possible matches, not plagiarism findings. "
    "Check each source yourself."
)

SAFE_URL_SCHEMES = frozenset({"http", "https"})

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")
_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Text hygiene
# ---------------------------------------------------------------------------


def clean(text: str) -> str:
    """Strip control and format characters except newlines and tabs.

    Removes things like NUL, ANSI escapes and bidi overrides that could
    disguise content in any output format.
    """
    out = []
    for ch in text:
        if ch in "\n\t":
            out.append(ch)
        elif unicodedata.category(ch) in ("Cc", "Cf"):
            continue
        else:
            out.append(ch)
    return "".join(out)


def one_line(text: str, limit: int | None = None) -> str:
    """Clean ``text``, collapse all whitespace to single spaces, and truncate."""
    flat = _WHITESPACE.sub(" ", clean(text)).strip()
    if limit is not None and len(flat) > limit:
        flat = flat[: limit - 1].rstrip() + "…"
    return flat


def safe_url(url: str) -> str | None:
    """Return a link-safe form of ``url``, or None if it must not be linked.

    Only absolute ``http``/``https`` URLs with a host are linkable. Characters
    that could break out of a markdown link target or an HTML attribute are
    percent-encoded.
    """
    candidate = url.strip()
    if not candidate or any(unicodedata.category(ch) in ("Cc", "Cf") for ch in candidate):
        return None
    if any(ch.isspace() for ch in candidate):
        return None
    try:
        parts = urlsplit(candidate)
    except ValueError:
        return None
    if parts.scheme.lower() not in SAFE_URL_SCHEMES or not parts.netloc:
        return None
    return quote(candidate, safe=":/?#@!$&'*+,;=%~-._")


# ---------------------------------------------------------------------------
# View dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceView:
    """One evidence item. ``href`` is None when the URL is not safe to link."""

    title: str
    url: str
    href: str | None
    excerpt: str
    is_primary: bool


@dataclass(frozen=True)
class FindingView:
    """One finding in must-fix, should-fix, polish or originality."""

    severity: str
    quote: str | None
    message: str
    suggestion: str | None
    sources: list[SourceView]
    heuristic: bool


@dataclass(frozen=True)
class TargetView:
    """A claim a rebuttal targets. ``text`` is None if it is not in the ledger."""

    claim_id: str
    text: str | None


@dataclass(frozen=True)
class RebuttalView:
    """One counter-case rebuttal."""

    argument: str
    strength: int
    target_kind: str
    targets: list[TargetView]
    sources: list[SourceView]


@dataclass(frozen=True)
class PolishGroup:
    """Copy edits sharing a location (a paragraph, or the whole document)."""

    label: str
    items: list[FindingView]


@dataclass(frozen=True)
class LedgerRow:
    """One claim in the appendix."""

    claim_id: str
    type: str
    importance: str
    verdict: str
    text: str
    note: str
    sources: list[SourceView]
    rebuttal_count: int


@dataclass(frozen=True)
class CountView:
    label: str
    count: int


@dataclass(frozen=True)
class UsageView:
    """Footer usage line pieces (pre-formatted)."""

    total_tokens: str
    input_tokens: str
    output_tokens: str
    llm_calls: str
    search_calls: str
    fetch_calls: str
    seconds: str | None


@dataclass(frozen=True)
class ReportView:
    """Everything a renderer needs, in design-doc section order."""

    verdict_line: str
    profile: str
    generated_at: str
    counts: list[CountView]
    must_fix: list[FindingView]
    counter_case: list[RebuttalView]
    should_fix: list[FindingView]
    polish: list[PolishGroup]
    originality: list[FindingView]
    ledger: list[LedgerRow]
    usage: UsageView
    notes: list[str] = field(default_factory=list)

    @property
    def total_findings(self) -> int:
        return sum(c.count for c in self.counts)


# ---------------------------------------------------------------------------
# Building the view
# ---------------------------------------------------------------------------


def build_view(report: Report, document: Document | None = None) -> ReportView:
    """Build the format-neutral view of ``report``.

    When ``document`` is given, quotes come from ``document.text`` at each
    span; otherwise claim text from the ledger is used where a finding links
    to a claim with the same span, and the quote is omitted otherwise.

    Raises ``ValueError`` if ``document`` is not the reviewed document.
    """
    if document is not None and document.id != report.document_id:
        raise ValueError(
            f"document {document.id!r} does not match report document {report.document_id!r}"
        )
    builder = _Builder(report, document)
    return ReportView(
        verdict_line=one_line(report.verdict_line),
        profile=report.profile.value.replace("_", " "),
        generated_at=report.generated_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        counts=_counts(report.counts),
        must_fix=[builder.finding(f) for f in report.must_fix],
        counter_case=[builder.rebuttal(r) for r in _top_rebuttals(report.counter_case)],
        should_fix=[builder.finding(f) for f in report.should_fix],
        polish=builder.polish_groups(report.polish),
        originality=[builder.finding(f) for f in report.originality],
        ledger=[builder.ledger_row(e) for e in _ledger_order(report)],
        usage=_usage(report),
        notes=[n for n in (one_line(n) for n in report.notes) if n],
    )


def _counts(counts: dict[str, int]) -> list[CountView]:
    order = {s.label: -s.value for s in Severity}
    keys = sorted(counts, key=lambda k: (order.get(k, 1), k))
    return [
        CountView(label=one_line(k).replace("_", " "), count=counts[k])
        for k in keys
        if counts[k] > 0
    ]


def _top_rebuttals(rebuttals: list[Rebuttal]) -> list[Rebuttal]:
    # Stable sort: equal strengths keep the orchestrator's order.
    return sorted(rebuttals, key=lambda r: -r.strength)[:MAX_COUNTER_CASE]


def _ledger_order(report: Report) -> list[LedgerEntry]:
    return sorted(report.ledger.entries.values(), key=lambda e: (e.claim.span.start, e.claim.id))


def _usage(report: Report) -> UsageView:
    u = report.usage
    return UsageView(
        total_tokens=f"{u.total_tokens:,}",
        input_tokens=f"{u.input_tokens:,}",
        output_tokens=f"{u.output_tokens:,}",
        llm_calls=f"{u.llm_calls:,}",
        search_calls=f"{u.search_calls:,}",
        fetch_calls=f"{u.fetch_calls:,}",
        seconds=f"{u.seconds:.1f}" if u.seconds > 0 else None,
    )


def _source(evidence: Evidence) -> SourceView:
    url = one_line(evidence.url)
    return SourceView(
        title=one_line(evidence.title) or url,
        url=url,
        href=safe_url(evidence.url),
        excerpt=one_line(evidence.excerpt, MAX_EXCERPT_CHARS),
        is_primary=evidence.is_primary,
    )


class _Builder:
    def __init__(self, report: Report, document: Document | None) -> None:
        self.report = report
        self.document = document

    def quote_for(self, span: Span | None, claim_ids: list[str]) -> str | None:
        if span is not None and self.document is not None:
            if span.is_valid_for(self.document.text) and span.end > span.start:
                return one_line(span.text_of(self.document.text), MAX_QUOTE_CHARS) or None
            return None
        entries = self.report.ledger.entries
        for cid in claim_ids:
            entry = entries.get(cid)
            if entry is not None and (span is None or entry.claim.span == span):
                return one_line(entry.claim.text, MAX_QUOTE_CHARS) or None
        return None

    def finding(self, finding: Finding) -> FindingView:
        suggestion = one_line(finding.suggestion) if finding.suggestion else None
        return FindingView(
            severity=finding.severity.label.replace("_", " "),
            quote=self.quote_for(finding.span, finding.claim_ids),
            message=one_line(finding.message),
            suggestion=suggestion or None,
            sources=[_source(e) for e in finding.evidence],
            heuristic=finding.heuristic,
        )

    def claim_text(self, claim_id: str) -> str | None:
        entry = self.report.ledger.entries.get(claim_id)
        if entry is None:
            return None
        claim = entry.claim
        if self.document is not None and claim.span.is_valid_for(self.document.text):
            text = claim.span.text_of(self.document.text)
        else:
            text = claim.text
        return one_line(text, MAX_QUOTE_CHARS) or None

    def rebuttal(self, rebuttal: Rebuttal) -> RebuttalView:
        return RebuttalView(
            argument=one_line(rebuttal.argument),
            strength=rebuttal.strength,
            target_kind=rebuttal.target.value,
            targets=[
                TargetView(claim_id=one_line(cid), text=self.claim_text(cid))
                for cid in rebuttal.target_claim_ids
            ],
            sources=[_source(e) for e in rebuttal.evidence],
        )

    def paragraph_of(self, span: Span) -> int | None:
        if self.document is None or not span.is_valid_for(self.document.text):
            return None
        # Paragraphs are blocks separated by blank lines; leading blank lines
        # do not count as a break.
        before = self.document.text[: span.start].lstrip()
        return len(_PARAGRAPH_BREAK.findall(before)) + 1

    def polish_groups(self, findings: list[Finding]) -> list[PolishGroup]:
        if not findings:
            return []
        if self.document is None:
            return [PolishGroup(label="Copy edits", items=[self.finding(f) for f in findings])]
        by_para: dict[int, list[Finding]] = {}
        whole: list[Finding] = []
        for f in findings:
            para = self.paragraph_of(f.span) if f.span is not None else None
            if para is None:
                whole.append(f)
            else:
                by_para.setdefault(para, []).append(f)
        groups = [
            PolishGroup(
                label=f"Paragraph {para}",
                items=[
                    self.finding(f)
                    for f in sorted(items, key=lambda f: f.span.start if f.span else 0)
                ],
            )
            for para, items in sorted(by_para.items())
        ]
        if whole:
            groups.append(
                PolishGroup(label="Whole document", items=[self.finding(f) for f in whole])
            )
        return groups

    def ledger_row(self, entry: LedgerEntry) -> LedgerRow:
        claim = entry.claim
        note = one_line(entry.verdict_note)
        if entry.debate is not None:
            ruling = one_line(entry.debate.ruling)
            note = f"{note} Debate ruling: {ruling}".strip() if ruling else note
        return LedgerRow(
            claim_id=one_line(claim.id),
            type=claim.type.value,
            importance=f"{claim.importance:.1f}",
            verdict=_verdict_label(entry.verdict),
            text=self.claim_text(claim.id) or "",
            note=note,
            sources=[_source(e) for e in entry.verdict_evidence],
            rebuttal_count=len(entry.rebuttals),
        )


def _verdict_label(verdict: Verdict) -> str:
    return verdict.value
