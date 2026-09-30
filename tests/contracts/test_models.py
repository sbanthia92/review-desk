import pytest
from pydantic import TypeAdapter, ValidationError

from reviewdesk.contracts import (
    PROFILE_AGENTS,
    AddClaim,
    AddRebuttal,
    AddResearchStep,
    AgentName,
    AgentResult,
    Claim,
    ClaimLedger,
    ClaimType,
    Document,
    Evidence,
    ExecutionPlan,
    Finding,
    LedgerUpdate,
    Profile,
    Rebuttal,
    Report,
    ResearchAction,
    ResearchStep,
    SetVerdict,
    Severity,
    Span,
    Usage,
    Verdict,
)
from reviewdesk.testing.fakes import OPINION_DOC, sample_documents, sample_ledger, sample_report


def test_document_from_text_counts_words():
    doc = Document.from_text("one two  three\nfour")
    assert doc.word_count == 4
    assert doc.id.startswith("doc_")


def test_span_rejects_reversed_range():
    with pytest.raises(ValidationError):
        Span(start=5, end=2)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0, 5), (5, 10), False),
        ((0, 6), (5, 10), True),
        ((3, 4), (0, 10), True),
        ((2, 2), (0, 10), False),
    ],
)
def test_span_overlap(a, b, expected):
    assert Span(start=a[0], end=a[1]).overlaps(Span(start=b[0], end=b[1])) is expected


def test_severity_order_matches_design():
    ordered = sorted(Severity, reverse=True)
    assert [s.label for s in ordered] == [
        "factual_error",
        "unsupported",
        "strong_rebuttal",
        "structure",
        "style",
        "consider",
    ]


def test_verified_requires_evidence():
    with pytest.raises(ValidationError):
        SetVerdict(claim_id="c", verdict=Verdict.VERIFIED)
    ledger = sample_ledger()
    with pytest.raises(ValidationError):
        ledger.set_verdict("claim_thesis", Verdict.VERIFIED, [])


def test_rebuttal_needs_a_target():
    with pytest.raises(ValidationError):
        Rebuttal(target_claim_ids=[], argument="x", strength=3)


def test_sample_ledger_spans_round_trip():
    ledger = sample_ledger()
    for entry in ledger.entries.values():
        assert entry.claim.span.text_of(OPINION_DOC.text) == entry.claim.text


def test_sample_documents():
    docs = sample_documents()
    assert len(docs) == 5
    assert len({d.id for d in docs}) == 5


def test_copy_edit_links_to_overlapping_claim():
    ledger = sample_ledger()
    assert ledger.get("claim_debate").copy_edit_ids == ["find_their"]
    ledger.remove_finding("find_their")
    assert ledger.get("claim_debate").copy_edit_ids == []


def test_copy_edit_added_before_claim_is_linked():
    ledger = ClaimLedger()
    ledger.add_finding(
        Finding(
            id="f1",
            agent=AgentName.COPYEDIT,
            severity=Severity.STYLE,
            span=Span(start=2, end=4),
            message="m",
        )
    )
    ledger.add_claim(
        Claim(
            id="c1",
            text="abcdef",
            span=Span(start=0, end=6),
            type=ClaimType.FACTUAL,
            importance=0.5,
        )
    )
    assert ledger.get("c1").copy_edit_ids == ["f1"]


def test_findings_overlapping():
    ledger = sample_ledger()
    span = ledger.get("claim_debate").claim.span
    assert [f.id for f in ledger.findings_overlapping(span)] == ["find_their"]
    assert ledger.findings_overlapping(Span(start=0, end=1)) == []


def test_add_rebuttal_attaches_to_every_target():
    ledger = sample_ledger()
    assert [r.id for r in ledger.get("claim_fatigue").rebuttals] == ["reb_fatigue"]
    assert {r.id for r in ledger.get("claim_thesis").rebuttals} == {"reb_fatigue", "reb_copy"}
    with pytest.raises(KeyError):
        ledger.add_rebuttal(Rebuttal(target_claim_ids=["nope"], argument="x", strength=1))


def test_claims_sorted_by_importance_and_thesis():
    ledger = sample_ledger()
    factual = ledger.claims(ClaimType.FACTUAL)
    assert [c.id for c in factual] == ["claim_goals", "claim_points", "claim_fatigue"]
    thesis = ledger.thesis()
    assert thesis is not None and thesis.id == "claim_thesis"
    assert ClaimLedger().thesis() is None


def test_apply_every_update_kind():
    ledger = ClaimLedger()
    claim = Claim(
        id="c1", text="x", span=Span(start=0, end=1), type=ClaimType.FACTUAL, importance=0.5
    )
    ev = Evidence(url="https://a", title="A", excerpt="x")
    updates: list[LedgerUpdate] = [
        AddClaim(claim=claim),
        SetVerdict(claim_id="c1", verdict=Verdict.VERIFIED, evidence=[ev], confidence=0.9),
        AddRebuttal(rebuttal=Rebuttal(target_claim_ids=["c1"], argument="no", strength=2)),
        AddResearchStep(
            claim_id="c1",
            step=ResearchStep(
                agent="factcheck", iteration=1, action=ResearchAction.SEARCH, input="q"
            ),
        ),
    ]
    ledger.apply_result(
        AgentResult(
            agent="t",
            ledger_updates=updates,
            findings=[Finding(agent="t", severity=Severity.STYLE, message="m")],
        )
    )
    entry = ledger.get("c1")
    assert entry.verdict is Verdict.VERIFIED
    assert entry.verdict_confidence == 0.9
    assert len(entry.rebuttals) == 1
    assert len(entry.research_trail) == 1
    assert len(ledger.findings) == 1


def test_ledger_update_union_round_trips_json():
    adapter = TypeAdapter(list[LedgerUpdate])
    updates: list[LedgerUpdate] = [
        AddClaim(
            claim=Claim(text="x", span=Span(start=0, end=1), type=ClaimType.THESIS, importance=1)
        ),
        SetVerdict(claim_id="c", verdict=Verdict.WRONG),
    ]
    assert adapter.validate_json(adapter.dump_json(updates)) == updates


def test_research_iteration_capped_at_three():
    with pytest.raises(ValidationError):
        ResearchStep(agent="a", iteration=4, action=ResearchAction.SEARCH, input="q")


def test_duplicate_claim_rejected():
    ledger = sample_ledger()
    with pytest.raises(ValueError):
        ledger.add_claim(ledger.get("claim_thesis").claim)


def test_usage_addition():
    total = Usage(input_tokens=1, output_tokens=2, search_calls=1) + Usage(
        input_tokens=3, llm_calls=1, seconds=1.5
    )
    assert total.total_tokens == 6
    assert total.search_calls == 1 and total.llm_calls == 1 and total.seconds == 1.5


def test_profile_agents_match_design():
    assert AgentName.ORIGINALITY in PROFILE_AGENTS[Profile.OPINION]
    assert AgentName.STRUCTURE not in PROFILE_AGENTS[Profile.OPINION]
    assert AgentName.STRUCTURE in PROFILE_AGENTS[Profile.DESIGN_DOC]
    assert AgentName.ORIGINALITY not in PROFILE_AGENTS[Profile.DESIGN_DOC]
    assert Profile.AUTO not in PROFILE_AGENTS


def test_plan_budget_for_falls_back_to_default():
    plan = ExecutionPlan(profile=Profile.OPINION, agents=[AgentName.FACTCHECK])
    assert plan.budget_for("missing").max_iterations == 3


def test_report_json_round_trip():
    report = sample_report()
    assert Report.model_validate_json(report.model_dump_json()) == report


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        Span.model_validate({"start": 0, "end": 1, "extra": True})
