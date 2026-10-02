"""Normalise a ``Report`` into scoreable items.

A report shows the author two kinds of things: findings (``must_fix``,
``should_fix``, ``polish``, ``originality``) and rebuttals (``counter_case``
plus every rebuttal in the ledger appendix). Both become ``ReportItem``s.

Each item gets a set of **roles** saying what kind of problem it reports. A
role comes from the finding's severity and from every contributing agent
(``agent`` plus ``merged_from``), so a merged fact + copy-edit finding carries
both the ``wrong`` and ``style`` roles. Rebuttal spans are the spans of the
claims they target (looked up in the report's ledger).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from reviewdesk.contracts import AgentName, Evidence, Finding, Rebuttal, Report, Severity, Span


class ItemKind(StrEnum):
    """Whether an item came from a finding or a rebuttal."""

    FINDING = "finding"
    REBUTTAL = "rebuttal"


class Role(StrEnum):
    """What kind of problem an item reports."""

    WRONG = "wrong"
    UNSUPPORTED = "unsupported"
    REBUTTAL = "rebuttal"
    ORIGINALITY = "originality"
    STRUCTURE = "structure"
    STYLE = "style"
    CONSIDER = "consider"


SEVERITY_ROLE: dict[Severity, Role] = {
    Severity.FACTUAL_ERROR: Role.WRONG,
    Severity.UNSUPPORTED: Role.UNSUPPORTED,
    Severity.STRONG_REBUTTAL: Role.REBUTTAL,
    Severity.STRUCTURE: Role.STRUCTURE,
    Severity.STYLE: Role.STYLE,
    Severity.CONSIDER: Role.CONSIDER,
}
"""Role implied by a finding's severity."""

AGENT_ROLE: dict[str, Role] = {
    AgentName.DEVILS_ADVOCATE: Role.REBUTTAL,
    AgentName.COPYEDIT: Role.STYLE,
    AgentName.STRUCTURE: Role.STRUCTURE,
    AgentName.ORIGINALITY: Role.ORIGINALITY,
}
"""Role implied by a contributing agent (the fact-checker's comes from severity)."""

FINDING_SECTIONS = ("must_fix", "unverified", "should_fix", "polish", "originality")
"""Report sections that hold findings, in report order."""


@dataclass(frozen=True)
class ReportItem:
    """One scoreable thing shown to the author.

    ``spans`` is empty for document-level findings and for rebuttals whose
    target claims are not in the ledger. ``source`` is the original model.
    """

    id: str
    kind: ItemKind
    section: str
    agents: frozenset[str]
    severity: Severity
    roles: frozenset[Role]
    spans: tuple[Span, ...]
    text: str
    evidence: tuple[Evidence, ...]
    source: Finding | Rebuttal

    def overlaps(self, span: Span) -> bool:
        """True if any of the item's spans shares a character with ``span``."""
        return any(s.overlaps(span) for s in self.spans)


def finding_agents(finding: Finding) -> frozenset[str]:
    """The producing agent plus every agent merged into the finding."""
    return frozenset({finding.agent, *finding.merged_from})


def finding_roles(finding: Finding) -> frozenset[Role]:
    """Roles of a finding: from its severity, its agents and ``heuristic``."""
    roles = {SEVERITY_ROLE[finding.severity]}
    roles.update(AGENT_ROLE[a] for a in finding_agents(finding) if a in AGENT_ROLE)
    if finding.heuristic:
        roles.add(Role.ORIGINALITY)
    if AgentName.FACTCHECK in finding_agents(finding) and finding.severity is Severity.CONSIDER:
        # "Could not verify": the fact-checker found no source for the claim,
        # which is how a planted unsupported claim shows up.
        roles.add(Role.UNSUPPORTED)
    return frozenset(roles)


def finding_item(finding: Finding, section: str) -> ReportItem:
    """Wrap one finding."""
    return ReportItem(
        id=finding.id,
        kind=ItemKind.FINDING,
        section=section,
        agents=finding_agents(finding),
        severity=finding.severity,
        roles=finding_roles(finding),
        spans=(finding.span,) if finding.span is not None else (),
        text=finding.message,
        evidence=tuple(finding.evidence),
        source=finding,
    )


def rebuttal_item(rebuttal: Rebuttal, report: Report, section: str) -> ReportItem:
    """Wrap one rebuttal; its spans are the targeted claims' spans."""
    spans = tuple(
        report.ledger.entries[cid].claim.span
        for cid in rebuttal.target_claim_ids
        if cid in report.ledger.entries
    )
    return ReportItem(
        id=rebuttal.id,
        kind=ItemKind.REBUTTAL,
        section=section,
        agents=frozenset({AgentName.DEVILS_ADVOCATE.value}),
        severity=Severity.STRONG_REBUTTAL if rebuttal.has_evidence else Severity.CONSIDER,
        roles=frozenset({Role.REBUTTAL}),
        spans=spans,
        text=rebuttal.argument,
        evidence=tuple(rebuttal.evidence),
        source=rebuttal,
    )


def report_items(report: Report) -> list[ReportItem]:
    """Every finding and rebuttal the report shows, de-duplicated by id."""
    items: list[ReportItem] = []
    seen: set[str] = set()
    for section in FINDING_SECTIONS:
        findings: list[Finding] = getattr(report, section)
        for finding in findings:
            if finding.id not in seen:
                seen.add(finding.id)
                items.append(finding_item(finding, section))
    for rebuttal in all_rebuttals(report):
        if rebuttal.id not in seen:
            seen.add(rebuttal.id)
            in_case = any(r.id == rebuttal.id for r in report.counter_case)
            items.append(rebuttal_item(rebuttal, report, "counter_case" if in_case else "ledger"))
    return items


def all_rebuttals(report: Report) -> list[Rebuttal]:
    """``counter_case`` rebuttals, then the rest from the ledger, by id."""
    found: dict[str, Rebuttal] = {r.id: r for r in report.counter_case}
    for entry in report.ledger.entries.values():
        for rebuttal in entry.rebuttals:
            found.setdefault(rebuttal.id, rebuttal)
    return list(found.values())


def devils_advocate_findings(report: Report) -> list[Finding]:
    """Findings the devil's advocate contributed to, in sections or the ledger.

    Rebuttal findings have no report section of their own, so the ledger's
    ``findings`` is where their final severity is visible.
    """
    found: dict[str, Finding] = {}
    for section in FINDING_SECTIONS:
        for f in getattr(report, section):
            found.setdefault(f.id, f)
    for f in report.ledger.findings.values():
        found.setdefault(f.id, f)
    return [f for f in found.values() if AgentName.DEVILS_ADVOCATE in finding_agents(f)]
