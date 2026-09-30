"""Conflict-resolution guardrails, as pure functions over the claim ledger.

Every rule takes a ``ClaimLedger``, never mutates it, and returns a
``Resolution``: a new ledger plus the ``Decision``s taken (IDs and reasons
only, never document text, so they are safe to trace). The rules, from
DESIGN.md "Conflict resolution rules":

1. ``drop_copy_edits_on_flagged_spans``: a copy edit touching a span flagged
   wrong or unsupported is dropped; the fact finding wins.
2. ``discard_fact_rebuttals_on_verified``: a rebuttal attacking a verified
   claim is kept only if it targets the interpretation; a fact-targeted one is
   discarded (or narrowed to its non-verified targets), with its finding.
3. ``merge_same_span_findings``: findings from two or more agents on the same
   span become one finding with every reason (``merged_from``).
4. ``downgrade_unsourced_rebuttals``: a rebuttal with no retrieved source is
   downgraded to ``Severity.CONSIDER``.
5. ``rank_findings``: factual_error > unsupported > strong_rebuttal >
   structure > style > consider.

``resolve_conflicts`` runs them in a fixed order (rebuttal findings are
ensured first, then 2, 4, 1, 3), so downgrades and drops happen before merges.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

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
from reviewdesk.orchestrator.labels import agent_rank, label

FLAGGED_VERDICTS = frozenset({Verdict.WRONG, Verdict.UNSUPPORTED})
"""Verdicts whose claim spans win over copy edits (rule 1)."""

FACT_SEVERITIES = frozenset({Severity.FACTUAL_ERROR, Severity.UNSUPPORTED})
"""Severities of fact findings (rule 1, and the report's "must fix")."""


@dataclass(frozen=True)
class Decision:
    """One guardrail decision: which rule, what it did, to which IDs, and why.

    ``ids`` are finding, rebuttal or claim IDs; ``reason`` never quotes the
    document.
    """

    rule: str
    action: str
    ids: tuple[str, ...]
    reason: str


@dataclass
class Resolution:
    """A rule's output: the new ledger and the decisions taken."""

    ledger: ClaimLedger
    decisions: list[Decision] = field(default_factory=list)


def rebuttal_finding_id(rebuttal_id: str) -> str:
    """ID of the finding that reports a rebuttal (the devil's advocate's convention)."""
    return f"find_{rebuttal_id}"


def all_rebuttals(ledger: ClaimLedger) -> list[Rebuttal]:
    """Every distinct rebuttal in the ledger (by ID), in ledger order."""
    seen: dict[str, Rebuttal] = {}
    for entry in ledger.entries.values():
        for rebuttal in entry.rebuttals:
            seen.setdefault(rebuttal.id, rebuttal)
    return list(seen.values())


def _anchor_span(ledger: ClaimLedger, claim_ids: Iterable[str]) -> Span | None:
    claims = [ledger.entries[cid].claim for cid in claim_ids if cid in ledger.entries]
    if not claims:
        return None
    return max(claims, key=lambda c: (c.importance, -c.span.start)).span


def unsourced_severity(rebuttal: Rebuttal) -> Severity:
    """Severity for a rebuttal's finding: strong if sourced, else consider."""
    return Severity.STRONG_REBUTTAL if rebuttal.has_evidence else Severity.CONSIDER


# ---------------------------------------------------------------------------
# Rebuttal findings
# ---------------------------------------------------------------------------


def ensure_rebuttal_findings(ledger: ClaimLedger) -> Resolution:
    """Add a finding for every rebuttal that has none (ID ``find_<rebuttal id>``).

    The devil's advocate normally emits these itself; this covers rebuttals
    that arrived without one so every rebuttal is counted exactly once.
    """
    out = ledger.model_copy(deep=True)
    decisions: list[Decision] = []
    for rebuttal in all_rebuttals(out):
        fid = rebuttal_finding_id(rebuttal.id)
        if fid in out.findings:
            continue
        out.add_finding(
            Finding(
                id=fid,
                agent=AgentName.DEVILS_ADVOCATE,
                severity=unsourced_severity(rebuttal),
                span=_anchor_span(out, rebuttal.target_claim_ids),
                message=rebuttal.argument,
                evidence=list(rebuttal.evidence),
                claim_ids=list(rebuttal.target_claim_ids),
            )
        )
        decisions.append(
            Decision("rebuttal_findings", "created", (fid,), "rebuttal had no finding")
        )
    return Resolution(out, decisions)


# ---------------------------------------------------------------------------
# Rule 1: copy edit vs fact finding
# ---------------------------------------------------------------------------


def flagged_spans(ledger: ClaimLedger) -> list[Span]:
    """Spans flagged wrong or unsupported, by verdict or by fact finding."""
    spans = [e.claim.span for e in ledger.entries.values() if e.verdict in FLAGGED_VERDICTS]
    spans += [
        f.span
        for f in ledger.findings.values()
        if f.span is not None and f.severity in FACT_SEVERITIES and f.agent != AgentName.COPYEDIT
    ]
    return spans


def drop_copy_edits_on_flagged_spans(ledger: ClaimLedger) -> Resolution:
    """Rule 1: drop copy edits overlapping a span flagged wrong or unsupported."""
    out = ledger.model_copy(deep=True)
    spans = flagged_spans(out)
    decisions: list[Decision] = []
    for finding in list(out.findings.values()):
        if finding.agent != AgentName.COPYEDIT or finding.span is None:
            continue
        span = finding.span
        if any(span.overlaps(flagged) for flagged in spans):
            out.remove_finding(finding.id)
            decisions.append(
                Decision(
                    "copy_edit_vs_fact",
                    "dropped",
                    (finding.id,),
                    "copy edit touches a span flagged wrong or unsupported",
                )
            )
    return Resolution(out, decisions)


# ---------------------------------------------------------------------------
# Rule 2: rebuttal vs verified claim
# ---------------------------------------------------------------------------


def _replace_rebuttal(out: ClaimLedger, rebuttal_id: str, new: Rebuttal | None) -> None:
    """Replace (or remove, if ``new`` is None) a rebuttal on every entry."""
    for cid, entry in out.entries.items():
        kept: list[Rebuttal] = []
        for existing in entry.rebuttals:
            if existing.id != rebuttal_id:
                kept.append(existing)
            elif new is not None and cid in new.target_claim_ids:
                kept.append(new)
        entry.rebuttals = kept


def discard_fact_rebuttals_on_verified(ledger: ClaimLedger) -> Resolution:
    """Rule 2: a rebuttal on a verified claim survives only if it targets the interpretation.

    A fact-targeted rebuttal is discarded with its finding when every target is
    verified. When only some targets are verified, it is narrowed to the
    others (its finding's claim IDs and anchor span follow).
    """
    out = ledger.model_copy(deep=True)
    decisions: list[Decision] = []
    for rebuttal in all_rebuttals(out):
        if rebuttal.target is not RebuttalTarget.FACT:
            continue
        verified = [
            cid
            for cid in rebuttal.target_claim_ids
            if cid in out.entries and out.entries[cid].verdict is Verdict.VERIFIED
        ]
        if not verified:
            continue
        remaining = [cid for cid in rebuttal.target_claim_ids if cid not in verified]
        fid = rebuttal_finding_id(rebuttal.id)
        if not remaining:
            _replace_rebuttal(out, rebuttal.id, None)
            out.remove_finding(fid)
            decisions.append(
                Decision(
                    "rebuttal_vs_verified",
                    "discarded",
                    (rebuttal.id, *verified),
                    "rebuttal attacks the fact of a verified claim",
                )
            )
            continue
        narrowed = rebuttal.model_copy(update={"target_claim_ids": remaining})
        _replace_rebuttal(out, rebuttal.id, narrowed)
        finding = out.findings.get(fid)
        if finding is not None:
            claim_ids = [cid for cid in finding.claim_ids if cid not in verified] or remaining
            verified_spans = [out.entries[cid].claim.span for cid in verified]
            span = finding.span
            if span is None or span in verified_spans:
                span = _anchor_span(out, remaining)
            out.findings[fid] = finding.model_copy(update={"claim_ids": claim_ids, "span": span})
        decisions.append(
            Decision(
                "rebuttal_vs_verified",
                "narrowed",
                (rebuttal.id, *verified),
                "fact-targeted rebuttal dropped from verified claims",
            )
        )
    return Resolution(out, decisions)


# ---------------------------------------------------------------------------
# Rule 3: same span, several agents
# ---------------------------------------------------------------------------


def _winner_key(finding: Finding) -> tuple[int, int, str]:
    return (-int(finding.severity), agent_rank(finding.agent), finding.id)


def union_evidence(groups: Iterable[Iterable[Evidence]]) -> list[Evidence]:
    """Evidence from every group, deduplicated by URL, first occurrence kept."""
    seen: set[str] = set()
    out: list[Evidence] = []
    for group in groups:
        for evidence in group:
            if evidence.url not in seen:
                seen.add(evidence.url)
                out.append(evidence)
    return out


def merge_findings(findings: list[Finding]) -> Finding:
    """Merge findings on one span into one: higher severity wins, reasons joined.

    The winner keeps its ID, agent, span and severity; every other message is
    appended with its agent's label; evidence and claim IDs are unioned;
    ``merged_from`` lists every contributing agent (winner first).
    """
    ordered = sorted(findings, key=_winner_key)
    winner, others = ordered[0], ordered[1:]
    message = winner.message
    for other in others:
        message += f" Also flagged by {label(other.agent)}: {other.message}"
    agents: list[str] = []
    for finding in ordered:
        for name in [finding.agent, *finding.merged_from]:
            if name not in agents:
                agents.append(name)
    claim_ids: list[str] = []
    for finding in ordered:
        for cid in finding.claim_ids:
            if cid not in claim_ids:
                claim_ids.append(cid)
    suggestion = next((f.suggestion for f in ordered if f.suggestion), None)
    return winner.model_copy(
        update={
            "message": message,
            "evidence": union_evidence(f.evidence for f in ordered),
            "claim_ids": claim_ids,
            "suggestion": suggestion,
            "merged_from": agents,
        }
    )


def merge_same_span_findings(ledger: ClaimLedger) -> Resolution:
    """Rule 3: findings from two or more agents on an identical span become one.

    Only identical spans merge (overlapping-but-different spans stay separate).
    Document-level findings (no span) and heuristic findings never merge.
    """
    out = ledger.model_copy(deep=True)
    groups: dict[tuple[int, int], list[Finding]] = {}
    for finding in out.findings.values():
        if finding.span is None or finding.heuristic:
            continue
        groups.setdefault((finding.span.start, finding.span.end), []).append(finding)
    decisions: list[Decision] = []
    for group in groups.values():
        if len({f.agent for f in group}) < 2:
            continue
        merged = merge_findings(group)
        for finding in group:
            if finding.id != merged.id:
                out.remove_finding(finding.id)
        out.findings[merged.id] = merged
        decisions.append(
            Decision(
                "same_span_merge",
                "merged",
                tuple(f.id for f in sorted(group, key=_winner_key)),
                f"{len(group)} findings from {len(merged.merged_from)} agents on one span",
            )
        )
    return Resolution(out, decisions)


# ---------------------------------------------------------------------------
# Rule 4: rebuttal without a source
# ---------------------------------------------------------------------------


def downgrade_unsourced_rebuttals(ledger: ClaimLedger) -> Resolution:
    """Rule 4: a rebuttal with no retrieved source is downgraded to ``CONSIDER``.

    Applies to the finding of every evidence-free rebuttal and to any other
    devil's advocate finding that cites nothing.
    """
    out = ledger.model_copy(deep=True)
    targets = {rebuttal_finding_id(r.id) for r in all_rebuttals(out) if not r.has_evidence}
    decisions: list[Decision] = []
    for fid, finding in list(out.findings.items()):
        unsourced_da = finding.agent == AgentName.DEVILS_ADVOCATE and not finding.evidence
        if (fid in targets or unsourced_da) and finding.severity > Severity.CONSIDER:
            out.findings[fid] = finding.model_copy(update={"severity": Severity.CONSIDER})
            decisions.append(
                Decision(
                    "unsourced_rebuttal",
                    "downgraded",
                    (fid,),
                    "rebuttal cites no retrieved source",
                )
            )
    return Resolution(out, decisions)


# ---------------------------------------------------------------------------
# Rule 5: ranking
# ---------------------------------------------------------------------------


def rank_key(finding: Finding) -> tuple[int, int, int, str]:
    """Sort key: severity (highest first), then document order, then ID.

    Document-level findings (no span) come after spanned ones of equal severity.
    """
    has_span = finding.span is not None
    start = finding.span.start if finding.span is not None else 0
    return (-int(finding.severity), 0 if has_span else 1, start, finding.id)


def rank_findings(findings: Iterable[Finding]) -> list[Finding]:
    """Rule 5: factual_error > unsupported > strong_rebuttal > structure > style > consider."""
    return sorted(findings, key=rank_key)


def rank_rebuttals(rebuttals: Iterable[Rebuttal]) -> list[Rebuttal]:
    """Rebuttals best first: evidence-backed first, then by strength, then ID."""
    return sorted(rebuttals, key=lambda r: (not r.has_evidence, -r.strength, r.id))


# ---------------------------------------------------------------------------
# All rules
# ---------------------------------------------------------------------------


def resolve_conflicts(ledger: ClaimLedger) -> Resolution:
    """Apply every guardrail in order and return the resolved ledger.

    Order: ensure rebuttal findings, rule 2 (verified claims), rule 4
    (downgrade), rule 1 (copy edits), rule 3 (merge). Findings in the result
    are stored in rank order (rule 5).
    """
    decisions: list[Decision] = []
    current = ledger
    for rule in (
        ensure_rebuttal_findings,
        discard_fact_rebuttals_on_verified,
        downgrade_unsourced_rebuttals,
        drop_copy_edits_on_flagged_spans,
        merge_same_span_findings,
    ):
        step = rule(current)
        current = step.ledger
        decisions.extend(step.decisions)
    current.findings = {f.id: f for f in rank_findings(current.findings.values())}
    return Resolution(current, decisions)
