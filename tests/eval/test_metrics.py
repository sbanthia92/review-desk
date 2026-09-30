"""Metrics on hand-built reports: matching, precision, citations, conflicts."""

from __future__ import annotations

from eval.dataset import ConflictRule, SeededDefect
from eval.items import ItemKind, Role, report_items
from eval.metrics import (
    ConflictStatus,
    check_conflict,
    evidence_wellformed,
    item_matches,
    score_report,
    span_matches,
)
from reviewdesk.contracts import (
    AgentName,
    Claim,
    ClaimLedger,
    ClaimType,
    Finding,
    Profile,
    Rebuttal,
    RebuttalTarget,
    Report,
    Severity,
    Span,
    Usage,
    Verdict,
)
from tests.eval.helpers import StubJudge, ev, span, tiny_seeded


def _finding(agent: str, severity: Severity, quote: str | None, fid: str, **kw: object) -> Finding:
    return Finding.model_validate(
        {
            "id": fid,
            "agent": agent,
            "severity": severity,
            "span": span(quote) if quote else None,
            "message": f"{agent} says so",
            **kw,
        }
    )


def _report(ledger: ClaimLedger | None = None, **sections: object) -> Report:
    return Report.model_validate(
        {
            "document_id": "tiny",
            "profile": Profile.OPINION,
            "verdict_line": "v",
            "ledger": ledger or ClaimLedger(),
            **sections,
        }
    )


def _claim(quote: str, cid: str, type_: ClaimType = ClaimType.SUPPORTING) -> Claim:
    return Claim(id=cid, text=quote, span=span(quote), type=type_, importance=0.5)


WRONG = "Liverpool won the 2019-20 league with 105 points."
UNSUP = "Pressing teams concede 50% fewer shots."
WEAK = "If one pressing team won, pressing always wins."
BORROWED = "It is not from the benevolence of the butcher that we expect our dinner."
GRAM = "The players was tired."
MERGE = "Their is no debate left"
VERIFIED = "The club was founded in 1892."


# -- matching -------------------------------------------------------------------


def test_span_match_needs_overlap_and_rejects_catch_all() -> None:
    defect = Span(start=100, end=120)
    assert span_matches(Span(start=110, end=115), defect)
    assert span_matches(Span(start=90, end=130), defect)
    assert not span_matches(Span(start=120, end=130), defect)  # touching, no overlap
    assert not span_matches(Span(start=0, end=500), defect)  # catch-all span
    assert span_matches(Span(start=60, end=160), defect)  # within 20 + 80 chars


def test_role_decides_which_defect_types_an_item_can_catch() -> None:
    seeded = tiny_seeded()
    wrong, unsup = seeded.defect("wrong"), seeded.defect("unsup")
    fact = report_items(
        _report(must_fix=[_finding("factcheck", Severity.FACTUAL_ERROR, WRONG, "f1")])
    )[0]
    assert fact.roles == {Role.WRONG}
    assert item_matches(fact, wrong)
    style = report_items(_report(polish=[_finding("copyedit", Severity.STYLE, WRONG, "f2")]))[0]
    assert not item_matches(style, wrong)
    as_wrong = report_items(
        _report(must_fix=[_finding("factcheck", Severity.FACTUAL_ERROR, UNSUP, "f3")])
    )[0]
    assert item_matches(as_wrong, unsup)  # flagged as wrong still catches unsupported
    as_unsup = report_items(
        _report(must_fix=[_finding("factcheck", Severity.UNSUPPORTED, WRONG, "f4")])
    )[0]
    assert not item_matches(as_unsup, wrong)  # but not the other way round


def test_merged_finding_carries_every_contributor_role() -> None:
    merged = _finding(
        "structure", Severity.STRUCTURE, GRAM, "m1", merged_from=["structure", "copyedit"]
    )
    item = report_items(_report(should_fix=[merged]))[0]
    assert {Role.STRUCTURE, Role.STYLE} <= item.roles
    assert item_matches(item, tiny_seeded().defect("gram"))


def test_rebuttal_spans_come_from_the_ledger() -> None:
    ledger = ClaimLedger()
    ledger.add_claim(_claim(WEAK, "c_weak"))
    rebuttal = Rebuttal(
        id="r1", target_claim_ids=["c_weak"], argument="Outlier.", evidence=[ev()], strength=4
    )
    ledger.add_rebuttal(rebuttal)
    items = report_items(_report(ledger, counter_case=[rebuttal]))
    assert len(items) == 1  # counter_case and ledger copies are de-duplicated
    item = items[0]
    assert item.kind is ItemKind.REBUTTAL and item.section == "counter_case"
    assert item.severity is Severity.STRONG_REBUTTAL
    assert item_matches(item, tiny_seeded().defect("weak"))


def test_originality_by_heuristic_flag() -> None:
    f = _finding("baseline", Severity.CONSIDER, BORROWED, "o1", heuristic=True)
    item = report_items(_report(originality=[f]))[0]
    assert item_matches(item, tiny_seeded().defect("borrowed"))


# -- whole-report scoring --------------------------------------------------------


def _full_report() -> Report:
    ledger = ClaimLedger()
    ledger.add_claim(_claim(WEAK, "c_weak"))
    rebuttal = Rebuttal(
        id="r1", target_claim_ids=["c_weak"], argument="Outlier.", evidence=[ev()], strength=4
    )
    ledger.add_rebuttal(rebuttal)
    return _report(
        ledger,
        must_fix=[
            _finding("factcheck", Severity.FACTUAL_ERROR, WRONG, "f_wrong", evidence=[ev()]),
            _finding("factcheck", Severity.UNSUPPORTED, UNSUP, "f_unsup"),
        ],
        counter_case=[rebuttal],
        polish=[
            _finding("copyedit", Severity.STYLE, GRAM, "f_gram"),
            _finding("copyedit", Severity.STYLE, "# Pressing wins titles", "f_fp"),
        ],
        originality=[
            _finding(
                "originality",
                Severity.CONSIDER,
                BORROWED,
                "f_orig",
                heuristic=True,
                evidence=[ev("ftp://bad", "x"), ev("https://example.org/b", "")],
            )
        ],
        usage=Usage(input_tokens=100, output_tokens=50, search_calls=3, llm_calls=2),
    )


async def test_score_report_offline() -> None:
    score = await score_report(tiny_seeded(), _full_report(), pipeline="p", seconds=1.5)
    assert all(o.caught for o in score.defects)
    assert score.items == 6 and score.items_matched == 5
    assert score.items_judged == 0 and not score.judged
    # f_wrong + r1 share (url, excerpt): scored once; two malformed originality sources.
    assert score.citations == 3 and score.citations_wellformed == 1
    assert score.usage.total_tokens == 150 and score.seconds == 1.5
    assert {c.defect_id for c in score.conflicts} == {"c_copy", "c_verified", "c_merge", "c_free"}


async def test_score_report_with_judge() -> None:
    judge = StubJudge(real=False, supports=True, score=5)
    score = await score_report(tiny_seeded(), _full_report(), judge=judge)
    assert score.judged
    assert score.items_judged == 1 and score.items_judged_real == 0  # only f_fp is judged
    assert "real:f_fp" in judge.calls
    assert score.citations_judged == 3 and score.citations_supported == 1  # malformed = False
    weak = next(o for o in score.defects if o.defect_id == "weak")
    assert weak.rebuttal_score == 5


async def test_judge_abstentions_leave_denominators() -> None:
    judge = StubJudge(real=None, supports=None, score=None)
    score = await score_report(tiny_seeded(), _full_report(), judge=judge)
    assert score.items_judged == 0
    assert score.citations_judged == 2  # the two malformed ones are judged False offline
    assert next(o for o in score.defects if o.defect_id == "weak").rebuttal_score is None


async def test_missed_defects_on_empty_report() -> None:
    score = await score_report(tiny_seeded(), _report())
    assert not any(o.caught for o in score.defects) and score.items == 0
    assert all(c.status is ConflictStatus.NOT_TRIGGERED for c in score.conflicts)


def test_evidence_wellformed() -> None:
    assert evidence_wellformed(ev())
    assert not evidence_wellformed(ev(excerpt="  "))
    assert not evidence_wellformed(ev(url="javascript:alert(1)"))
    assert not evidence_wellformed(ev(url="https://"))


# -- conflicts -------------------------------------------------------------------


def _conflict(rule: ConflictRule) -> SeededDefect:
    return next(d for d in tiny_seeded().defects if d.rule is rule)


def test_copyedit_yields_to_fact() -> None:
    d = _conflict(ConflictRule.COPYEDIT_YIELDS_TO_FACT)
    fact = _finding("factcheck", Severity.FACTUAL_ERROR, WRONG, "f1")
    edit = _finding("copyedit", Severity.STYLE, "105 points", "e1")
    assert check_conflict(d, _report(must_fix=[fact])).status is ConflictStatus.RESOLVED
    bad = check_conflict(d, _report(must_fix=[fact], polish=[edit]))
    assert bad.status is ConflictStatus.VIOLATED and "e1" in bad.detail
    merged = fact.model_copy(update={"merged_from": ["factcheck", "copyedit"]})
    assert check_conflict(d, _report(must_fix=[merged])).status is ConflictStatus.RESOLVED
    assert check_conflict(d, _report(polish=[edit])).status is ConflictStatus.NOT_TRIGGERED
    ledger = ClaimLedger()
    ledger.add_claim(_claim(WRONG, "c1", ClaimType.FACTUAL))
    ledger.set_verdict("c1", Verdict.WRONG, [ev()])
    assert check_conflict(d, _report(ledger, polish=[edit])).status is ConflictStatus.VIOLATED


def test_rebuttal_on_verified_fact() -> None:
    d = _conflict(ConflictRule.REBUTTAL_ON_VERIFIED_FACT)

    def ledger_with(target: RebuttalTarget | None, verdict: Verdict) -> ClaimLedger:
        ledger = ClaimLedger()
        ledger.add_claim(_claim(VERIFIED, "c1", ClaimType.FACTUAL))
        if verdict is Verdict.VERIFIED:
            ledger.set_verdict("c1", verdict, [ev()])
        if target is not None:
            ledger.add_rebuttal(
                Rebuttal(id="r1", target_claim_ids=["c1"], argument="a", strength=2, target=target)
            )
        return ledger

    check = check_conflict
    assert check(d, _report(ledger_with(None, Verdict.VERIFIED))).status is ConflictStatus.RESOLVED
    interp = ledger_with(RebuttalTarget.INTERPRETATION, Verdict.VERIFIED)
    assert check(d, _report(interp)).status is ConflictStatus.RESOLVED
    fact = ledger_with(RebuttalTarget.FACT, Verdict.VERIFIED)
    assert check(d, _report(fact)).status is ConflictStatus.VIOLATED
    unchecked = ledger_with(RebuttalTarget.FACT, Verdict.UNCHECKED)
    assert check(d, _report(unchecked)).status is ConflictStatus.NOT_TRIGGERED


def test_same_span_merge() -> None:
    d = _conflict(ConflictRule.SAME_SPAN_MERGE)
    edit = _finding("copyedit", Severity.STYLE, "Their is", "e1")
    struct = _finding("structure", Severity.STRUCTURE, MERGE, "s1")
    merged = struct.model_copy(update={"merged_from": ["structure", "copyedit"]})
    assert check_conflict(d, _report(should_fix=[merged])).status is ConflictStatus.RESOLVED
    both = _report(should_fix=[struct], polish=[edit])
    assert check_conflict(d, both).status is ConflictStatus.VIOLATED
    assert check_conflict(d, _report(polish=[edit])).status is ConflictStatus.NOT_TRIGGERED


def test_evidence_free_rebuttal() -> None:
    d = _conflict(ConflictRule.EVIDENCE_FREE_REBUTTAL)
    ledger = ClaimLedger()
    ledger.add_claim(_claim(WEAK, "c_weak"))
    high = _finding(
        AgentName.DEVILS_ADVOCATE, Severity.STRONG_REBUTTAL, None, "da1", claim_ids=["c_weak"]
    )
    ledger.add_finding(high)
    assert check_conflict(d, _report(ledger)).status is ConflictStatus.VIOLATED
    ledger.findings["da1"] = high.model_copy(update={"severity": Severity.CONSIDER})
    assert check_conflict(d, _report(ledger)).status is ConflictStatus.RESOLVED
    ledger.findings["da1"] = high.model_copy(update={"evidence": [ev()]})
    assert check_conflict(d, _report(ledger)).status is ConflictStatus.NOT_TRIGGERED
