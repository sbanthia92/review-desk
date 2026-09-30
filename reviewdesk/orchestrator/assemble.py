"""Report assembly from a resolved ledger (DESIGN.md "Report structure")."""

from __future__ import annotations

from reviewdesk.contracts import (
    AgentName,
    ClaimLedger,
    Document,
    Finding,
    Profile,
    Rebuttal,
    Report,
    Severity,
    Usage,
)
from reviewdesk.orchestrator.conflicts import (
    FACT_SEVERITIES,
    all_rebuttals,
    rank_findings,
    rank_rebuttals,
)

COUNTER_CASE_SIZE = 3


def _is_originality(finding: Finding) -> bool:
    return finding.heuristic or finding.agent == AgentName.ORIGINALITY


def _has_copy_edit(finding: Finding) -> bool:
    return finding.agent == AgentName.COPYEDIT or AgentName.COPYEDIT in finding.merged_from


def counter_case(ledger: ClaimLedger, size: int = COUNTER_CASE_SIZE) -> list[Rebuttal]:
    """Top ``size`` surviving rebuttals: evidence-backed first, then by strength."""
    return rank_rebuttals(all_rebuttals(ledger))[:size]


def count_by_severity(findings: list[Finding]) -> dict[str, int]:
    """Counts keyed by ``Severity.label``, highest severity first, zeros omitted."""
    counts: dict[str, int] = {}
    for severity in sorted(Severity, reverse=True):
        n = sum(1 for f in findings if f.severity is severity)
        if n:
            counts[severity.label] = n
    return counts


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def verdict_line(findings: list[Finding], *, partial: bool = False) -> str:
    """One sentence on overall readiness."""
    errors = sum(1 for f in findings if f.severity is Severity.FACTUAL_ERROR)
    unsupported = sum(1 for f in findings if f.severity is Severity.UNSUPPORTED)
    rebuttals = sum(1 for f in findings if f.severity is Severity.STRONG_REBUTTAL)
    structure = sum(1 for f in findings if f.severity is Severity.STRUCTURE)
    if errors or unsupported:
        parts = []
        if errors:
            parts.append(_plural(errors, "factual error"))
        if unsupported:
            parts.append(_plural(unsupported, "unsupported claim"))
        line = f"Not ready: {' and '.join(parts)} to fix."
    elif rebuttals or structure:
        parts = []
        if rebuttals:
            parts.append(_plural(rebuttals, "strong counter-argument"))
        if structure:
            parts.append(_plural(structure, "structure issue"))
        line = f"Nearly ready: no factual problems found; address {' and '.join(parts)}."
    elif findings:
        line = "Ready: only minor suggestions."
    else:
        line = "Ready: no issues found."
    if partial:
        line += " Partial review: see notes."
    return line


def build_report(
    document: Document,
    profile: Profile,
    ledger: ClaimLedger,
    *,
    usage: Usage,
    notes: list[str],
    partial: bool = False,
) -> Report:
    """Assemble the ``Report`` from a ledger already run through the conflict rules.

    Sections: ``must_fix`` (factual errors, unsupported), ``counter_case`` (top 3
    rebuttals), ``should_fix`` (structure), ``polish`` (copy edits, including
    merged ones not already in a higher section), ``originality`` (heuristic
    findings). Rebuttal findings are reported through ``counter_case`` and
    ``counts``.
    """
    ranked = rank_findings(ledger.findings.values())
    must_fix: list[Finding] = []
    should_fix: list[Finding] = []
    polish: list[Finding] = []
    originality: list[Finding] = []
    for finding in ranked:
        if _is_originality(finding):
            originality.append(finding)
        elif finding.severity in FACT_SEVERITIES:
            must_fix.append(finding)
        elif finding.severity is Severity.STRUCTURE:
            should_fix.append(finding)
        elif finding.severity is Severity.STYLE or _has_copy_edit(finding):
            polish.append(finding)
    return Report(
        document_id=document.id,
        profile=profile,
        verdict_line=verdict_line(ranked, partial=partial),
        counts=count_by_severity(ranked),
        must_fix=must_fix,
        counter_case=counter_case(ledger),
        should_fix=should_fix,
        polish=polish,
        originality=originality,
        ledger=ledger,
        usage=usage,
        notes=list(dict.fromkeys(notes)),
    )
