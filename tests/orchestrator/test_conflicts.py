"""Table-driven tests for every conflict-resolution rule."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from reviewdesk.contracts import (
    AgentName,
    Finding,
    Rebuttal,
    RebuttalTarget,
    Severity,
    Span,
    Verdict,
)
from reviewdesk.orchestrator.conflicts import (
    discard_fact_rebuttals_on_verified,
    downgrade_unsourced_rebuttals,
    drop_copy_edits_on_flagged_spans,
    ensure_rebuttal_findings,
    merge_same_span_findings,
    rank_findings,
    rank_rebuttals,
    resolve_conflicts,
)
from reviewdesk.testing.fakes import sample_ledger
from tests.orchestrator.builders import (
    bare_ledger,
    claim_span,
    da_finding_for,
    ev,
    finding,
    rebuttal,
    sp,
    verify,
)

CE = AgentName.COPYEDIT
FC = AgentName.FACTCHECK
DA = AgentName.DEVILS_ADVOCATE
ST = AgentName.STRUCTURE
OR = AgentName.ORIGINALITY
FACT = RebuttalTarget.FACT
INTERP = RebuttalTarget.INTERPRETATION

# ---------------------------------------------------------------------------
# Rule 1: copy edit touching a span flagged wrong or unsupported is dropped
# ---------------------------------------------------------------------------


@dataclass
class CopyEditCase:
    name: str
    verdicts: dict[str, Verdict]
    copy_span: Span
    kept: bool
    extra: list[Finding] = field(default_factory=list)


def _goals_end() -> Span:
    s = claim_span("claim_goals")
    return Span(start=s.end, end=s.end + 5)


COPY_EDIT_CASES = [
    CopyEditCase("inside wrong claim", {"claim_goals": Verdict.WRONG}, sp("their press"), False),
    CopyEditCase(
        "inside unsupported claim",
        {"claim_fatigue": Verdict.UNSUPPORTED},
        sp("simply not"),
        False,
    ),
    CopyEditCase(
        "partial overlap with wrong claim",
        {"claim_goals": Verdict.WRONG},
        Span(start=claim_span("claim_goals").start - 5, end=claim_span("claim_goals").start + 3),
        False,
    ),
    CopyEditCase("on verified claim is kept", {"claim_points": Verdict.VERIFIED}, sp("99"), True),
    CopyEditCase("on unchecked claim is kept", {}, sp("their is"), True),
    CopyEditCase(
        "adjacent to wrong claim is kept", {"claim_goals": Verdict.WRONG}, _goals_end(), True
    ),
    CopyEditCase(
        "overlapping a fact finding with no verdict",
        {},
        sp("every club should copy"),
        False,
        extra=[
            finding(
                "find_fact",
                FC,
                Severity.UNSUPPORTED,
                sp("every club should copy the approach immediately"),
            )
        ],
    ),
    CopyEditCase(
        "overlapping a structure finding is kept",
        {},
        sp("every club should copy"),
        True,
        extra=[finding("find_st", ST, Severity.STRUCTURE, sp("every club should copy the"))],
    ),
]


@pytest.mark.parametrize("case", COPY_EDIT_CASES, ids=lambda c: c.name)
def test_rule1_copy_edit_vs_fact(case: CopyEditCase) -> None:
    ledger = bare_ledger()
    for cid, verdict in case.verdicts.items():
        evidence = [ev()] if verdict is Verdict.VERIFIED else []
        ledger.set_verdict(cid, verdict, evidence)
    for extra in case.extra:
        ledger.add_finding(extra)
    ledger.add_finding(finding("find_ce", CE, Severity.STYLE, case.copy_span))
    before = ledger.model_copy(deep=True)

    result = drop_copy_edits_on_flagged_spans(ledger)

    assert ("find_ce" in result.ledger.findings) is case.kept
    assert ledger == before, "rules must not mutate their input"
    if not case.kept:
        assert result.decisions[0].rule == "copy_edit_vs_fact"
        assert all("find_ce" not in e.copy_edit_ids for e in result.ledger.entries.values())
    for extra in case.extra:
        assert extra.id in result.ledger.findings


# ---------------------------------------------------------------------------
# Rule 2: rebuttal on a verified claim survives only if it targets interpretation
# ---------------------------------------------------------------------------


@dataclass
class VerifiedCase:
    name: str
    verified: list[str]
    reb: Rebuttal
    expect_targets: list[str] | None  # None -> discarded


VERIFIED_CASES = [
    VerifiedCase(
        "fact rebuttal on verified claim discarded",
        ["claim_points"],
        rebuttal("r1", ["claim_points"], target=FACT, evidence=[ev()]),
        None,
    ),
    VerifiedCase(
        "unsourced fact rebuttal on verified claim discarded",
        ["claim_points"],
        rebuttal("r1", ["claim_points"], target=FACT),
        None,
    ),
    VerifiedCase(
        "interpretation rebuttal on verified claim kept",
        ["claim_points"],
        rebuttal("r1", ["claim_points"], target=INTERP, evidence=[ev()]),
        ["claim_points"],
    ),
    VerifiedCase(
        "fact rebuttal on unverified claim kept",
        [],
        rebuttal("r1", ["claim_points"], target=FACT, evidence=[ev()]),
        ["claim_points"],
    ),
    VerifiedCase(
        "fact rebuttal on several claims narrowed to the unverified ones",
        ["claim_points"],
        rebuttal("r1", ["claim_goals", "claim_points"], target=FACT, evidence=[ev()]),
        ["claim_goals"],
    ),
    VerifiedCase(
        "fact rebuttal on several verified claims discarded",
        ["claim_points", "claim_goals"],
        rebuttal("r1", ["claim_goals", "claim_points"], target=FACT, evidence=[ev()]),
        None,
    ),
    VerifiedCase(
        "interpretation rebuttal on several claims, one verified, kept whole",
        ["claim_points"],
        rebuttal("r1", ["claim_thesis", "claim_points"], target=INTERP, evidence=[ev()]),
        ["claim_thesis", "claim_points"],
    ),
]


@pytest.mark.parametrize("case", VERIFIED_CASES, ids=lambda c: c.name)
def test_rule2_rebuttal_vs_verified(case: VerifiedCase) -> None:
    ledger = verify(bare_ledger(), *case.verified)
    ledger.add_rebuttal(case.reb)
    ledger.add_finding(da_finding_for(case.reb, ledger))
    before = ledger.model_copy(deep=True)

    result = discard_fact_rebuttals_on_verified(ledger)
    out = result.ledger

    assert ledger == before
    holders = {cid for cid, e in out.entries.items() if any(r.id == "r1" for r in e.rebuttals)}
    if case.expect_targets is None:
        assert holders == set()
        assert "find_r1" not in out.findings
        assert result.decisions[0].action == "discarded"
        return
    assert holders == set(case.expect_targets)
    for cid in holders:
        [kept] = [r for r in out.entries[cid].rebuttals if r.id == "r1"]
        assert kept.target_claim_ids == case.expect_targets
    kept_finding = out.findings["find_r1"]
    assert kept_finding.claim_ids == case.expect_targets
    anchors = {out.entries[cid].claim.span for cid in case.expect_targets}
    assert kept_finding.span in anchors


# ---------------------------------------------------------------------------
# Rule 3: two agents flag the same span -> one finding with both reasons
# ---------------------------------------------------------------------------


@dataclass
class MergeCase:
    name: str
    findings: list[Finding]
    expect_ids: set[str]
    merged_id: str | None = None
    merged_from: list[str] = field(default_factory=list)
    merged_severity: Severity | None = None


SPAN_A = sp("Liverpool won the 2019-20 Premier League with 99 points")
SPAN_B = sp("Liverpool won the 2019-20 Premier League")

MERGE_CASES = [
    MergeCase(
        "fact and rebuttal on same span merge, fact wins",
        [
            finding("f_fc", FC, Severity.FACTUAL_ERROR, SPAN_A, evidence=[ev("https://a")]),
            finding("f_da", DA, Severity.STRONG_REBUTTAL, SPAN_A, evidence=[ev("https://b")]),
        ],
        {"f_fc"},
        "f_fc",
        [FC, DA],
        Severity.FACTUAL_ERROR,
    ),
    MergeCase(
        "higher severity wins regardless of order",
        [
            finding("a_ce", CE, Severity.STYLE, SPAN_A),
            finding("z_st", ST, Severity.STRUCTURE, SPAN_A),
        ],
        {"z_st"},
        "z_st",
        [ST, CE],
        Severity.STRUCTURE,
    ),
    MergeCase(
        "three agents merge into one",
        [
            finding("f1", CE, Severity.STYLE, SPAN_A),
            finding("f2", DA, Severity.STRONG_REBUTTAL, SPAN_A),
            finding("f3", FC, Severity.UNSUPPORTED, SPAN_A),
        ],
        {"f3"},
        "f3",
        [FC, DA, CE],
        Severity.UNSUPPORTED,
    ),
    MergeCase(
        "same agent twice on one span does not merge",
        [finding("f1", CE, Severity.STYLE, SPAN_A), finding("f2", CE, Severity.STYLE, SPAN_A)],
        {"f1", "f2"},
    ),
    MergeCase(
        "overlapping but not identical spans do not merge",
        [
            finding("f1", FC, Severity.FACTUAL_ERROR, SPAN_A),
            finding("f2", DA, Severity.STRONG_REBUTTAL, SPAN_B),
        ],
        {"f1", "f2"},
    ),
    MergeCase(
        "document-level findings do not merge",
        [finding("f1", ST, Severity.STRUCTURE), finding("f2", DA, Severity.CONSIDER)],
        {"f1", "f2"},
    ),
    MergeCase(
        "heuristic originality findings do not merge",
        [
            finding("f1", CE, Severity.STYLE, SPAN_A),
            finding("f2", OR, Severity.CONSIDER, SPAN_A, heuristic=True),
        ],
        {"f1", "f2"},
    ),
]


@pytest.mark.parametrize("case", MERGE_CASES, ids=lambda c: c.name)
def test_rule3_same_span_merge(case: MergeCase) -> None:
    ledger = bare_ledger()
    for f in case.findings:
        ledger.add_finding(f)
    before = ledger.model_copy(deep=True)

    result = merge_same_span_findings(ledger)
    out = result.ledger

    assert ledger == before
    assert set(out.findings) == case.expect_ids
    if case.merged_id is None:
        assert result.decisions == []
        return
    merged = out.findings[case.merged_id]
    assert merged.merged_from == case.merged_from
    assert merged.severity is case.merged_severity
    for f in case.findings:
        assert f.message in merged.message, "every reason is kept"
    urls = {e.url for f in case.findings for e in f.evidence}
    assert {e.url for e in merged.evidence} == urls


def test_rule3_merge_unions_claims_and_dedupes_evidence() -> None:
    ledger = bare_ledger()
    ledger.add_finding(
        finding(
            "f1", FC, Severity.FACTUAL_ERROR, SPAN_A, evidence=[ev("https://a")], claim_ids=["c1"]
        )
    )
    ledger.add_finding(
        finding(
            "f2",
            DA,
            Severity.STRONG_REBUTTAL,
            SPAN_A,
            evidence=[ev("https://a"), ev("https://b")],
            claim_ids=["c1", "c2"],
            suggestion="Answer it.",
        )
    )
    merged = merge_same_span_findings(ledger).ledger.findings["f1"]
    assert [e.url for e in merged.evidence] == ["https://a", "https://b"]
    assert merged.claim_ids == ["c1", "c2"]
    assert merged.suggestion == "Answer it."
    assert merge_same_span_findings(merge_same_span_findings(ledger).ledger).decisions == []


# ---------------------------------------------------------------------------
# Rule 4: rebuttal with no retrieved source -> consider
# ---------------------------------------------------------------------------


@dataclass
class DowngradeCase:
    name: str
    reb: Rebuttal | None
    finding: Finding
    expect: Severity


DOWNGRADE_CASES = [
    DowngradeCase(
        "unsourced rebuttal finding downgraded",
        rebuttal("r1", ["claim_thesis"]),
        finding("find_r1", DA, Severity.STRONG_REBUTTAL, claim_span("claim_thesis")),
        Severity.CONSIDER,
    ),
    DowngradeCase(
        "sourced rebuttal keeps its severity",
        rebuttal("r1", ["claim_thesis"], evidence=[ev()]),
        finding(
            "find_r1", DA, Severity.STRONG_REBUTTAL, claim_span("claim_thesis"), evidence=[ev()]
        ),
        Severity.STRONG_REBUTTAL,
    ),
    DowngradeCase(
        "unsourced rebuttal on several claims downgraded",
        rebuttal("r1", ["claim_thesis", "claim_shots"]),
        finding("find_r1", DA, Severity.STRONG_REBUTTAL, claim_span("claim_thesis")),
        Severity.CONSIDER,
    ),
    DowngradeCase(
        "devil's advocate finding citing nothing downgraded",
        None,
        finding("f_da", DA, Severity.STRONG_REBUTTAL),
        Severity.CONSIDER,
    ),
    DowngradeCase(
        "fact finding without evidence untouched",
        None,
        finding("f_fc", FC, Severity.UNSUPPORTED, claim_span("claim_fatigue")),
        Severity.UNSUPPORTED,
    ),
    DowngradeCase(
        "already consider stays consider",
        rebuttal("r1", ["claim_thesis"]),
        finding("find_r1", DA, Severity.CONSIDER),
        Severity.CONSIDER,
    ),
]


@pytest.mark.parametrize("case", DOWNGRADE_CASES, ids=lambda c: c.name)
def test_rule4_unsourced_rebuttal_downgrade(case: DowngradeCase) -> None:
    ledger = bare_ledger()
    if case.reb is not None:
        ledger.add_rebuttal(case.reb)
    ledger.add_finding(case.finding)
    before = ledger.model_copy(deep=True)

    out = downgrade_unsourced_rebuttals(ledger).ledger

    assert ledger == before
    assert out.findings[case.finding.id].severity is case.expect


# ---------------------------------------------------------------------------
# Rule 5: ranking by severity
# ---------------------------------------------------------------------------

RANK_CASES = [
    (
        "severity order",
        [
            finding("style", CE, Severity.STYLE, sp("their is")),
            finding("consider", DA, Severity.CONSIDER, sp("High pressing")),
            finding("fe", FC, Severity.FACTUAL_ERROR, sp("their press")),
            finding("structure", ST, Severity.STRUCTURE),
            finding("reb", DA, Severity.STRONG_REBUTTAL, sp("Critics say")),
            finding("uns", FC, Severity.UNSUPPORTED, sp("That is simply not true")),
        ],
        ["fe", "uns", "reb", "structure", "style", "consider"],
    ),
    (
        "ties broken by document order, document-level last",
        [
            finding("late", FC, Severity.UNSUPPORTED, sp("That is simply not true")),
            finding("doc", ST, Severity.UNSUPPORTED),
            finding("early", FC, Severity.UNSUPPORTED, sp("Liverpool won")),
        ],
        ["early", "late", "doc"],
    ),
]


@pytest.mark.parametrize(("name", "findings", "expect"), RANK_CASES, ids=[c[0] for c in RANK_CASES])
def test_rule5_ranking(name: str, findings: list[Finding], expect: list[str]) -> None:
    assert [f.id for f in rank_findings(findings)] == expect
    assert [f.id for f in rank_findings(reversed(findings))] == expect


def test_rank_rebuttals_evidence_first_then_strength() -> None:
    rebs = [
        rebuttal("weak_sourced", ["claim_thesis"], evidence=[ev()], strength=2),
        rebuttal("strong_unsourced", ["claim_thesis"], strength=5),
        rebuttal("strong_sourced", ["claim_thesis"], evidence=[ev()], strength=4),
    ]
    assert [r.id for r in rank_rebuttals(rebs)] == [
        "strong_sourced",
        "weak_sourced",
        "strong_unsourced",
    ]


# ---------------------------------------------------------------------------
# Rebuttal findings and the full rule chain
# ---------------------------------------------------------------------------


def test_ensure_rebuttal_findings_creates_only_missing() -> None:
    ledger = sample_ledger()  # reb_fatigue (sourced) and reb_copy (unsourced), no findings
    ledger.add_finding(
        da_finding_for(ledger.get("claim_fatigue").rebuttals[0], ledger)
    )  # find_reb_fatigue exists already
    out = ensure_rebuttal_findings(ledger)
    assert [d.ids for d in out.decisions] == [("find_reb_copy",)]
    created = out.ledger.findings["find_reb_copy"]
    assert created.agent == DA
    assert created.severity is Severity.CONSIDER
    assert created.span == claim_span("claim_thesis")
    assert created.claim_ids == ["claim_thesis"]


def test_resolve_conflicts_on_sample_ledger() -> None:
    ledger = sample_ledger()
    ledger.add_finding(
        finding(
            "find_goals",
            FC,
            Severity.FACTUAL_ERROR,
            claim_span("claim_goals"),
            claim_ids=["claim_goals"],
        )
    )
    ledger.add_finding(finding("find_ce_goals", CE, Severity.STYLE, sp("their press")))
    fact_attack = rebuttal("reb_fact", ["claim_points"], target=FACT, evidence=[ev()])
    ledger.add_rebuttal(fact_attack)
    ledger.add_finding(da_finding_for(fact_attack, ledger))
    before = ledger.model_copy(deep=True)

    result = resolve_conflicts(ledger)
    out = result.ledger

    assert ledger == before
    assert "find_ce_goals" not in out.findings  # rule 1
    assert "find_their" in out.findings  # copy edit on unchecked claim kept
    assert "find_reb_fact" not in out.findings  # rule 2
    assert all(r.id != "reb_fact" for e in out.entries.values() for r in e.rebuttals)
    assert out.findings["find_reb_copy"].severity is Severity.CONSIDER  # created + rule 4
    assert out.findings["find_reb_fatigue"].severity is Severity.STRONG_REBUTTAL
    severities = [f.severity for f in out.findings.values()]
    assert severities == sorted(severities, reverse=True)  # rule 5
    rules = {d.rule for d in result.decisions}
    assert {"rebuttal_findings", "rebuttal_vs_verified", "copy_edit_vs_fact"} <= rules
    again = resolve_conflicts(out)
    assert again.ledger == out, "resolution is idempotent"
