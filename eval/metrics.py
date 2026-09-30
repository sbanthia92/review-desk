"""Score one ``Report`` against its answer key.

Matching rule (documented in ``eval/README.md``). An item (see
``eval.items``) **matches** a seeded defect when both hold:

1. **Role**: the item carries a role the defect type accepts
   (``ACCEPTED_ROLES``): ``wrong_fact`` → ``wrong``; ``unsupported`` →
   ``unsupported`` or ``wrong``; ``weak_argument`` → ``rebuttal`` (a rebuttal or
   a devil's-advocate finding); ``borrowed_sentence`` → ``originality``;
   ``grammar`` → ``style``. Roles come from severity and contributing agents,
   so merged findings count for every contributor.
2. **Span**: one of the item's spans overlaps the defect span by at least one
   character and is not a catch-all: its length is at most
   ``max(4 × defect length, defect length + 80)`` characters.

A defect is **caught** if any item matches it. An item is a **true positive**
if it matches any seeded defect; unmatched items go to the judge (if any) and
count as real only if it says so.

Conflicts are scored per rule (``check_conflict``): each seeded conflict is
``resolved``, ``violated`` or ``not_triggered`` (the situation the rule governs
never arose, e.g. no fact finding on the span). Accuracy is resolved over
resolved + violated.
"""

from __future__ import annotations

from enum import StrEnum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from eval.dataset import RECALL_TYPES, ConflictRule, DefectType, SeededDefect, SeededDocument
from eval.items import (
    ItemKind,
    ReportItem,
    Role,
    all_rebuttals,
    devils_advocate_findings,
    report_items,
)
from eval.judge import EvalJudge
from reviewdesk.contracts import (
    Evidence,
    Rebuttal,
    RebuttalTarget,
    Report,
    Severity,
    Span,
    Usage,
    Verdict,
)

ACCEPTED_ROLES: dict[DefectType, frozenset[Role]] = {
    DefectType.WRONG_FACT: frozenset({Role.WRONG}),
    DefectType.UNSUPPORTED: frozenset({Role.UNSUPPORTED, Role.WRONG}),
    DefectType.WEAK_ARGUMENT: frozenset({Role.REBUTTAL}),
    DefectType.BORROWED_SENTENCE: frozenset({Role.ORIGINALITY}),
    DefectType.GRAMMAR: frozenset({Role.STYLE}),
}
"""Item roles that count as catching each defect type."""

SPAN_SLACK_CHARS = 80
SPAN_MAX_FACTOR = 4
MAX_REBUTTAL_CANDIDATES = 3
"""Rebuttals judged per weak-argument defect (the best score is kept)."""


class ConflictStatus(StrEnum):
    """Outcome of one seeded conflict."""

    RESOLVED = "resolved"
    VIOLATED = "violated"
    NOT_TRIGGERED = "not_triggered"


class DefectOutcome(BaseModel):
    """Whether one seeded defect was caught, and by which items."""

    model_config = ConfigDict(extra="forbid")

    defect_id: str
    type: DefectType
    caught: bool
    matched_item_ids: list[str] = Field(default_factory=list)
    rebuttal_score: int | None = None
    """Best judge score (1–5) among matched rebuttals; weak arguments only."""


class ConflictOutcome(BaseModel):
    """How one seeded conflict was resolved."""

    model_config = ConfigDict(extra="forbid")

    defect_id: str
    rule: ConflictRule
    status: ConflictStatus
    detail: str = ""


class DocScore(BaseModel):
    """Scores for one pipeline run on one document."""

    model_config = ConfigDict(extra="forbid")

    pipeline: str
    doc_id: str
    repeat: int = 0
    error: str | None = None
    defects: list[DefectOutcome] = Field(default_factory=list)
    items: int = 0
    items_matched: int = 0
    items_judged: int = 0
    """Unmatched items the judge gave a decision on."""
    items_judged_real: int = 0
    citations: int = 0
    citations_wellformed: int = 0
    citations_judged: int = 0
    citations_supported: int = 0
    conflicts: list[ConflictOutcome] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    seconds: float = 0.0
    judged: bool = False
    """True when a judge was available for this score."""


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def span_matches(item_span: Span, defect_span: Span) -> bool:
    """Overlap by one character or more, and the item span is not a catch-all."""
    if not item_span.overlaps(defect_span):
        return False
    defect_len = defect_span.end - defect_span.start
    limit = max(SPAN_MAX_FACTOR * defect_len, defect_len + SPAN_SLACK_CHARS)
    return item_span.end - item_span.start <= limit


def item_matches(item: ReportItem, defect: SeededDefect) -> bool:
    """True if ``item`` catches ``defect`` (role and span rules above)."""
    accepted = ACCEPTED_ROLES.get(defect.type)
    if accepted is None or not (item.roles & accepted):
        return False
    return any(span_matches(s, defect.located) for s in item.spans)


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def evidence_wellformed(evidence: Evidence) -> bool:
    """Cheap offline check: non-empty excerpt and an absolute http(s) URL."""
    parsed = urlparse(evidence.url.strip())
    return (
        bool(evidence.excerpt.strip())
        and parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
    )


def citations(seeded: SeededDocument, report: Report) -> list[tuple[str, Evidence]]:
    """Every cited source with the statement it is cited for, de-duplicated.

    Sources come from findings and rebuttals the report shows, then from
    ledger verdicts (verified or wrong claims). A (url, excerpt) pair cited
    twice is scored once, for the first statement.
    """
    text = seeded.text
    pairs: list[tuple[str, Evidence]] = []
    seen: set[tuple[str, str]] = set()

    def add(statement: str, evidence: Evidence) -> None:
        key = (evidence.url, evidence.excerpt)
        if key not in seen:
            seen.add(key)
            pairs.append((statement, evidence))

    for item in report_items(report):
        quoted = " / ".join(s.text_of(text) for s in item.spans)
        if item.kind is ItemKind.REBUTTAL:
            statement = f'Rebuttal of "{quoted}": {item.text}'
        else:
            statement = f'{item.severity.label} finding on "{quoted}": {item.text}'
        for ev in item.evidence:
            add(statement, ev)
    for entry in report.ledger.entries.values():
        if entry.verdict in (Verdict.VERIFIED, Verdict.WRONG):
            statement = f'The claim "{entry.claim.text}" is {entry.verdict.value}.'
            if entry.verdict_note:
                statement += f" {entry.verdict_note}"
            for ev in entry.verdict_evidence:
                add(statement, ev)
    return pairs


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------


def check_conflict(defect: SeededDefect, report: Report) -> ConflictOutcome:
    """Score one seeded conflict against the report (see module docstring)."""
    if defect.rule is None:
        raise ValueError(f"defect {defect.id} is not a conflict")
    span = defect.located
    items = [i for i in report_items(report) if i.kind is ItemKind.FINDING and i.overlaps(span)]
    status, detail = _CHECKS[defect.rule](span, items, report)
    return ConflictOutcome(defect_id=defect.id, rule=defect.rule, status=status, detail=detail)


_Check = tuple[ConflictStatus, str]


def _copyedit_yields(span: Span, items: list[ReportItem], report: Report) -> _Check:
    """Copy edit on a span flagged wrong/unsupported must be dropped.

    Triggered when a fact finding overlaps the span or an overlapping claim
    has verdict wrong/unsupported. Violated by any standalone style finding
    overlapping the span (a style role on an item that is not itself a fact
    finding); a copy edit merged into the fact finding is accepted.
    """
    fact_roles = {Role.WRONG, Role.UNSUPPORTED}
    flagged = any(i.roles & fact_roles for i in items) or any(
        e.verdict in (Verdict.WRONG, Verdict.UNSUPPORTED)
        for e in report.ledger.claims_overlapping(span)
    )
    if not flagged:
        return ConflictStatus.NOT_TRIGGERED, "span not flagged wrong or unsupported"
    stray = [i.id for i in items if Role.STYLE in i.roles and not (i.roles & fact_roles)]
    if stray:
        return ConflictStatus.VIOLATED, f"copy edits kept on a flagged span: {stray}"
    return ConflictStatus.RESOLVED, "fact finding wins; no copy edit on the span"


def _rebuttal_on_verified(span: Span, items: list[ReportItem], report: Report) -> _Check:
    """A rebuttal of a verified claim survives only if it targets interpretation.

    Triggered when a claim overlapping the span is verified. Violated by any
    rebuttal in the report (counter case or ledger) that targets such a claim
    with ``target == fact``.
    """
    verified = {
        e.claim.id for e in report.ledger.claims_overlapping(span) if e.verdict is Verdict.VERIFIED
    }
    if not verified:
        return ConflictStatus.NOT_TRIGGERED, "no verified claim on the span"
    bad = [
        r.id
        for r in all_rebuttals(report)
        if r.target is RebuttalTarget.FACT and verified.intersection(r.target_claim_ids)
    ]
    if bad:
        return ConflictStatus.VIOLATED, f"fact rebuttals kept against a verified claim: {bad}"
    return ConflictStatus.RESOLVED, "no fact rebuttal against the verified claim"


def _same_span_merge(span: Span, items: list[ReportItem], report: Report) -> _Check:
    """Two agents flagging the same span must be merged into one finding.

    Triggered when findings overlapping the span come from two or more agents
    in total. Resolved when a single finding carries all of them; violated when
    they are still separate findings.
    """
    contributors = set().union(*(i.agents for i in items)) if items else set()
    if len(contributors) < 2:
        return ConflictStatus.NOT_TRIGGERED, "fewer than two agents flagged the span"
    if len(items) == 1:
        return ConflictStatus.RESOLVED, f"merged: {sorted(contributors)}"
    return ConflictStatus.VIOLATED, f"{len(items)} separate findings on one span"


def _evidence_free_rebuttal(span: Span, items: list[ReportItem], report: Report) -> _Check:
    """A rebuttal with no retrieved source must be downgraded to ``consider``.

    Looks at devil's-advocate findings (in sections or the ledger) that overlap
    the span or link to a claim overlapping it. Triggered when one of them
    cites no evidence; violated if any such finding ranks above ``consider``.
    """
    claim_ids = {e.claim.id for e in report.ledger.claims_overlapping(span)}
    linked = [
        f
        for f in devils_advocate_findings(report)
        if (f.span is not None and f.span.overlaps(span)) or claim_ids.intersection(f.claim_ids)
    ]
    free = [f for f in linked if not f.evidence]
    if not free:
        return ConflictStatus.NOT_TRIGGERED, "no evidence-free rebuttal finding on the span"
    high = [f.id for f in free if f.severity > Severity.CONSIDER]
    if high:
        return ConflictStatus.VIOLATED, f"evidence-free rebuttals above consider: {high}"
    return ConflictStatus.RESOLVED, "evidence-free rebuttals downgraded to consider"


_CHECKS = {
    ConflictRule.COPYEDIT_YIELDS_TO_FACT: _copyedit_yields,
    ConflictRule.REBUTTAL_ON_VERIFIED_FACT: _rebuttal_on_verified,
    ConflictRule.SAME_SPAN_MERGE: _same_span_merge,
    ConflictRule.EVIDENCE_FREE_REBUTTAL: _evidence_free_rebuttal,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _rebuttal_candidates(items: list[ReportItem]) -> list[ReportItem]:
    """Best rebuttal candidates first: real rebuttals with evidence, by strength."""

    def key(item: ReportItem) -> tuple[int, int, int]:
        strength = item.source.strength if isinstance(item.source, Rebuttal) else 0
        return (int(item.kind is ItemKind.REBUTTAL), int(bool(item.evidence)), strength)

    return sorted(items, key=key, reverse=True)[:MAX_REBUTTAL_CANDIDATES]


def _candidate_text(item: ReportItem) -> str:
    sources = "; ".join(f"{e.title} <{e.url}>: {e.excerpt[:200]}" for e in item.evidence[:3])
    return item.text + (f"\nSources: {sources}" if sources else "")


async def score_report(
    seeded: SeededDocument,
    report: Report,
    *,
    pipeline: str = "",
    repeat: int = 0,
    seconds: float = 0.0,
    error: str | None = None,
    judge: EvalJudge | None = None,
) -> DocScore:
    """Score ``report`` against ``seeded``'s answer key.

    With ``judge=None`` only the offline metrics are filled (matched-only
    precision, well-formed citations, conflicts, recall, cost); the judged
    counters stay 0 and ``judged`` is False.
    """
    doc = seeded.document
    items = report_items(report)
    matched_ids: set[str] = set()
    outcomes: list[DefectOutcome] = []
    for defect in seeded.of_type(*RECALL_TYPES):
        hits = [i for i in items if item_matches(i, defect)]
        matched_ids.update(i.id for i in hits)
        outcome = DefectOutcome(
            defect_id=defect.id,
            type=defect.type,
            caught=bool(hits),
            matched_item_ids=[i.id for i in hits],
        )
        if judge is not None and hits and defect.type is DefectType.WEAK_ARGUMENT:
            scores = [
                await judge.rebuttal_strength(
                    doc, defect.quote, defect.known_rebuttal or "", _candidate_text(c)
                )
                for c in _rebuttal_candidates(hits)
            ]
            valid = [s for s in scores if s is not None]
            outcome.rebuttal_score = max(valid) if valid else None
        outcomes.append(outcome)

    score = DocScore(
        pipeline=pipeline,
        doc_id=seeded.id,
        repeat=repeat,
        error=error,
        defects=outcomes,
        items=len(items),
        items_matched=len(matched_ids),
        usage=report.usage,
        seconds=seconds,
        judged=judge is not None,
    )

    if judge is not None:
        for item in items:
            if item.id in matched_ids:
                continue
            real = await judge.is_real(doc, item)
            if real is not None:
                score.items_judged += 1
                score.items_judged_real += int(real)

    for statement, evidence in citations(seeded, report):
        score.citations += 1
        ok = evidence_wellformed(evidence)
        score.citations_wellformed += int(ok)
        if judge is not None:
            supported = await judge.supports(statement, evidence) if ok else False
            if supported is not None:
                score.citations_judged += 1
                score.citations_supported += int(supported)

    score.conflicts = [check_conflict(d, report) for d in seeded.of_type(DefectType.CONFLICT)]
    return score
