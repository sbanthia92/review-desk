"""Small builders shared by the orchestrator tests (all on ``OPINION_DOC``)."""

from __future__ import annotations

from collections.abc import Iterable

from reviewdesk.contracts import (
    AgentName,
    ClaimLedger,
    Evidence,
    Finding,
    Rebuttal,
    RebuttalTarget,
    Severity,
    Span,
    Verdict,
)
from reviewdesk.testing.fakes import FIXED_TIME, OPINION_DOC, sample_ledger, span_of

DOC = OPINION_DOC


def sp(quote: str) -> Span:
    """Span of ``quote`` in the opinion doc."""
    return span_of(DOC.text, quote)


def ev(url: str = "https://example.org/a", *, primary: bool = False) -> Evidence:
    return Evidence(
        url=url, title="Source", excerpt="An excerpt.", retrieved_at=FIXED_TIME, is_primary=primary
    )


def finding(
    id: str,
    agent: str,
    severity: Severity,
    span: Span | None = None,
    *,
    evidence: Iterable[Evidence] = (),
    claim_ids: Iterable[str] = (),
    heuristic: bool = False,
    message: str | None = None,
    suggestion: str | None = None,
) -> Finding:
    return Finding(
        id=id,
        agent=agent,
        severity=severity,
        span=span,
        message=message or f"{agent} says {id}",
        evidence=list(evidence),
        claim_ids=list(claim_ids),
        heuristic=heuristic,
        suggestion=suggestion,
    )


def rebuttal(
    id: str,
    targets: list[str],
    *,
    target: RebuttalTarget = RebuttalTarget.INTERPRETATION,
    evidence: Iterable[Evidence] = (),
    strength: int = 3,
) -> Rebuttal:
    return Rebuttal(
        id=id,
        target_claim_ids=targets,
        argument=f"argument {id}",
        evidence=list(evidence),
        strength=strength,
        target=target,
    )


def bare_ledger() -> ClaimLedger:
    """``sample_ledger()`` claims only: no verdicts, rebuttals or findings."""
    full = sample_ledger()
    ledger = ClaimLedger()
    for entry in full.entries.values():
        ledger.add_claim(entry.claim)
    return ledger


def claim_span(claim_id: str) -> Span:
    return sample_ledger().get(claim_id).claim.span


def verify(ledger: ClaimLedger, *claim_ids: str) -> ClaimLedger:
    for cid in claim_ids:
        ledger.set_verdict(cid, Verdict.VERIFIED, [ev(f"https://example.org/{cid}")])
    return ledger


def da_finding_for(reb: Rebuttal, ledger: ClaimLedger) -> Finding:
    """The finding the devil's advocate emits for ``reb`` (id ``find_<id>``)."""
    anchor = max(
        (ledger.get(cid).claim for cid in reb.target_claim_ids), key=lambda c: c.importance
    )
    return finding(
        f"find_{reb.id}",
        AgentName.DEVILS_ADVOCATE,
        Severity.STRONG_REBUTTAL if reb.has_evidence else Severity.CONSIDER,
        anchor.span,
        evidence=reb.evidence,
        claim_ids=reb.target_claim_ids,
    )
