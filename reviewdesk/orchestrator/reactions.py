"""Reaction-round helpers: who gets re-checked, who debates, and code rulings.

Pure functions only; ``Orchestrator`` does the calls. From DESIGN.md "Agents
react to each other":

- Re-check: the devil's advocate's evidence contradicts a claim marked
  verified (an evidence-backed rebuttal with ``target == FACT``) -> one
  re-check of that claim by the fact-checker.
- Debate: for the top 3 claims by importance where the two still disagree
  after re-checks, each side responds once and the orchestrator rules.
"""

from __future__ import annotations

from reviewdesk.contracts import (
    AgentName,
    ClaimLedger,
    DebatePosition,
    DebateRecord,
    Evidence,
    Finding,
    LedgerEntry,
    Rebuttal,
    RebuttalTarget,
    Severity,
    Verdict,
)
from reviewdesk.orchestrator.conflicts import union_evidence
from reviewdesk.orchestrator.schemas import RulingReply

DEBATE_LIMIT = 3
"""Debate rounds per job: the top 3 disputed claims by importance."""


def contradicting_rebuttals(entry: LedgerEntry) -> list[Rebuttal]:
    """Evidence-backed, fact-targeted rebuttals on a verified claim."""
    if entry.verdict is not Verdict.VERIFIED:
        return []
    return [r for r in entry.rebuttals if r.target is RebuttalTarget.FACT and r.has_evidence]


def counter_evidence(entry: LedgerEntry) -> list[Evidence]:
    """The devil's advocate's evidence against ``entry`` (deduplicated by URL)."""
    return union_evidence(r.evidence for r in contradicting_rebuttals(entry))


def recheck_candidates(ledger: ClaimLedger) -> dict[str, list[Evidence]]:
    """Verified, not-yet-rechecked claims contradicted by evidence -> that evidence.

    Ordered most important first.
    """
    out: dict[str, list[Evidence]] = {}
    for claim in ledger.claims():
        entry = ledger.entries[claim.id]
        if entry.rechecked:
            continue
        evidence = counter_evidence(entry)
        if evidence:
            out[claim.id] = evidence
    return out


def debate_candidates(ledger: ClaimLedger, limit: int = DEBATE_LIMIT) -> list[str]:
    """Claims still in dispute (verified yet contradicted), top ``limit`` by importance."""
    disputed = [
        c.id
        for c in ledger.claims()
        if ledger.entries[c.id].debate is None and counter_evidence(ledger.entries[c.id])
    ]
    return disputed[:limit]


def fallback_ruling(
    factcheck: DebatePosition | None, devils_advocate: DebatePosition | None
) -> RulingReply:
    """The code ruling when the LLM ruling fails.

    The devil's advocate wins only if the fact-checker concedes and the devil's
    advocate does not; otherwise the evidence-backed verified verdict stands.
    """
    fc_concedes = factcheck is not None and factcheck.concedes
    da_concedes = devils_advocate is not None and devils_advocate.concedes
    if fc_concedes and not da_concedes:
        return RulingReply(
            winner="devils_advocate",
            final_verdict="wrong",
            ruling="The fact-checker conceded to the counter-evidence.",
            reasoning="Code ruling: the fact-checker conceded; the claim is marked wrong.",
        )
    return RulingReply(
        winner="factcheck",
        final_verdict="verified",
        ruling="The verified verdict stands.",
        reasoning="Code ruling: the fact-checker did not concede, so its evidence-backed "
        "verdict stands.",
    )


def ruling_is_consistent(reply: RulingReply) -> bool:
    """A factcheck win keeps 'verified'; a devil's advocate win must overturn it."""
    if reply.winner == "factcheck":
        return reply.final_verdict == "verified"
    return reply.final_verdict in ("wrong", "unsupported")


def debate_record(
    reply: RulingReply,
    factcheck: DebatePosition | None,
    devils_advocate: DebatePosition | None,
) -> DebateRecord:
    """Build the ledger's ``DebateRecord`` from positions and a ruling."""
    return DebateRecord(
        factcheck_position=factcheck.argument if factcheck else "(no response)",
        devils_advocate_position=devils_advocate.argument if devils_advocate else "(no response)",
        ruling=" ".join(reply.ruling.split())[:500],
        reasoning=" ".join(reply.reasoning.split())[:1000],
        final_verdict=Verdict(reply.final_verdict),
    )


def apply_ruling(ledger: ClaimLedger, claim_id: str, record: DebateRecord) -> None:
    """Record the debate on the entry and apply an overturned verdict (mutates ``ledger``).

    When the devil's advocate wins, the verdict changes to ``wrong`` or
    ``unsupported`` with its counter-evidence, and an orchestrator finding
    reports it (keyed ``find_debate_<claim id>``).
    """
    entry = ledger.entries[claim_id]
    evidence = counter_evidence(entry)
    entry.debate = record
    if record.final_verdict is Verdict.VERIFIED:
        return
    ledger.set_verdict(
        claim_id,
        record.final_verdict,
        evidence,
        note=f"Overturned after debate: {record.ruling}"[:500],
    )
    severity = (
        Severity.FACTUAL_ERROR if record.final_verdict is Verdict.WRONG else Severity.UNSUPPORTED
    )
    ledger.add_finding(
        Finding(
            id=f"find_debate_{claim_id}",
            agent=AgentName.ORCHESTRATOR,
            severity=severity,
            span=entry.claim.span,
            message=f"After a debate, the counter-evidence prevailed. {record.reasoning}"[:1200],
            evidence=evidence,
            suggestion="Correct the claim or cite a source that settles it.",
            claim_ids=[claim_id],
        )
    )
