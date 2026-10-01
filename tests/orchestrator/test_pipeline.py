"""The full orchestrator pipeline on fakes: report, progress, plans, degradation, reactions."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

from reviewdesk import pipeline
from reviewdesk.contracts import (
    AddClaim,
    AddRebuttal,
    AgentName,
    AgentResult,
    Budget,
    DebatePosition,
    Document,
    Finding,
    LedgerUpdate,
    MissingSetup,
    Profile,
    ProgressEvent,
    ProviderError,
    RebuttalTarget,
    Report,
    ReviewContext,
    ReviewPipeline,
    SetVerdict,
    Severity,
    Verdict,
)
from reviewdesk.orchestrator import STEP_ORDER, AgentRegistry, Orchestrator
from reviewdesk.testing.fakes import (
    DESIGN_DOC,
    OPINION_DOC,
    FakeAgent,
    FakeDevilsAdvocate,
    FakeFactChecker,
    FakeFetcher,
    FakeLLM,
    FakeSearch,
    ProgressRecorder,
    sample_documents,
    sample_ledger,
)
from tests.orchestrator.builders import claim_span, da_finding_for, ev, finding, rebuttal, sp

FC = AgentName.FACTCHECK
DA = AgentName.DEVILS_ADVOCATE
CE = AgentName.COPYEDIT

_pipeline_conforms: ReviewPipeline = pipeline.run_review


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def claim_updates() -> list[LedgerUpdate]:
    return [AddClaim(claim=e.claim) for e in sample_ledger().entries.values()]


def extractor() -> FakeAgent:
    return FakeAgent(
        "extractor", ledger_updates=claim_updates(), progress=["chunk 1 of 2", "chunk 2 of 2"]
    )


def verdicts(**by_claim: Verdict) -> list[LedgerUpdate]:
    out: list[LedgerUpdate] = []
    for cid, verdict in by_claim.items():
        evidence = [ev(f"https://example.org/{cid}")] if verdict is Verdict.VERIFIED else []
        out.append(SetVerdict(claim_id=cid, verdict=verdict, evidence=evidence))
    return out


def standard_factchecker(**kwargs: Any) -> FakeFactChecker:
    return FakeFactChecker(
        ledger_updates=verdicts(
            claim_points=Verdict.VERIFIED,
            claim_goals=Verdict.WRONG,
            claim_fatigue=Verdict.UNSUPPORTED,
        ),
        findings=[
            finding(
                "find_goals",
                FC,
                Severity.FACTUAL_ERROR,
                claim_span("claim_goals"),
                evidence=[ev("https://example.org/goals")],
                claim_ids=["claim_goals"],
            ),
            finding(
                "find_fatigue",
                FC,
                Severity.UNSUPPORTED,
                claim_span("claim_fatigue"),
                claim_ids=["claim_fatigue"],
            ),
        ],
        progress=["Fact-checking claim 1 of 3"],
        **kwargs,
    )


REB_FATIGUE = rebuttal(
    "reb_fatigue", ["claim_thesis", "claim_fatigue"], evidence=[ev("https://ex.org/f")], strength=4
)
REB_COPY = rebuttal("reb_copy", ["claim_thesis"], strength=5)
REB_SHOTS = rebuttal("reb_shots", ["claim_shots"], evidence=[ev("https://ex.org/s")], strength=3)
REB_EXTRA = rebuttal("reb_extra", ["claim_debate"], evidence=[ev("https://ex.org/x")], strength=2)


def standard_advocate(**kwargs: Any) -> FakeDevilsAdvocate:
    ledger = sample_ledger()
    rebs = [REB_FATIGUE, REB_COPY, REB_SHOTS, REB_EXTRA]
    return FakeDevilsAdvocate(
        ledger_updates=[AddRebuttal(rebuttal=r) for r in rebs],
        # T5 emits find_<id> per rebuttal; leave reb_copy without one on purpose.
        findings=[da_finding_for(r, ledger) for r in rebs if r.id != "reb_copy"],
        **kwargs,
    )


def standard_agents() -> list[Any]:
    return [
        extractor(),
        standard_factchecker(),
        standard_advocate(),
        FakeAgent(
            "copyedit",
            findings=[
                finding("find_their", CE, Severity.STYLE, sp("their is"), suggestion="there is"),
                finding("find_ce_goals", CE, Severity.STYLE, sp("their press")),
            ],
        ),
        FakeAgent(
            "originality",
            findings=[
                finding(
                    "find_orig",
                    AgentName.ORIGINALITY,
                    Severity.CONSIDER,
                    sp("High pressing is the single"),
                    evidence=[ev("https://example.com/blog")],
                    heuristic=True,
                )
            ],
        ),
        FakeAgent(
            "structure",
            findings=[finding("find_structure", AgentName.STRUCTURE, Severity.STRUCTURE)],
        ),
    ]


def make_registry(agents: list[Any], llm: FakeLLM | None = None) -> AgentRegistry:
    return AgentRegistry.of(
        agents, llm=llm or FakeLLM(), search=FakeSearch(), fetcher=FakeFetcher()
    )


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Iterator[None]:
    pipeline.set_default_registry(None)
    yield
    pipeline.set_default_registry(None)


def assert_progress(rec: ProgressRecorder) -> None:
    percents = [e.percent for e in rec.events]
    assert percents == sorted(percents), "percent never goes backwards"
    assert all(0 <= p <= 100 for p in percents)
    steps = list(dict.fromkeys(rec.steps))
    assert steps == [s.value for s in STEP_ORDER], "every step, in order"
    for a, b in zip(rec.steps, rec.steps[1:], strict=False):
        assert [s.value for s in STEP_ORDER].index(a) <= [s.value for s in STEP_ORDER].index(b)
    assert rec.events[-1].step == "done"
    assert rec.events[-1].percent == 100.0


def all_findings(report: Report) -> list[Finding]:
    return list(report.ledger.findings.values())


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def test_full_fake_pipeline_produces_valid_report() -> None:
    rec = ProgressRecorder()
    report = await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, rec, registry=make_registry(standard_agents())
    )

    assert Report.model_validate(report.model_dump(mode="json")) == report
    assert report.document_id == OPINION_DOC.id
    assert report.profile is Profile.OPINION
    assert [f.id for f in report.must_fix] == ["find_goals", "find_fatigue"]
    assert [f.id for f in report.polish] == ["find_their"]  # rule 1 dropped find_ce_goals
    assert "find_ce_goals" not in report.ledger.findings
    assert [f.id for f in report.originality] == ["find_orig"]
    assert report.should_fix == []  # structure is not an opinion-profile agent
    assert [r.id for r in report.counter_case] == ["reb_fatigue", "reb_shots", "reb_extra"]
    assert report.ledger.findings["find_reb_copy"].severity is Severity.CONSIDER
    assert report.counts == {
        "factual_error": 1,
        "unsupported": 1,
        "strong_rebuttal": 3,
        "style": 1,
        "consider": 2,
    }
    assert sum(report.counts.values()) == len(report.ledger.findings)
    assert report.verdict_line.startswith("Not ready: 1 factual error and 1 unsupported claim")
    assert report.notes == []
    assert report.usage.seconds > 0
    assert_progress(rec)
    messages = [e.message for e in rec.events]
    assert "chunk 1 of 2" in messages
    assert "Fact-checking claim 1 of 3" in messages


async def test_agents_share_one_meter_and_ledger_snapshot() -> None:
    agents = standard_agents()
    fc, da, ce = agents[1], agents[2], agents[3]
    await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, ProgressRecorder(), registry=make_registry(agents)
    )
    extract_ctx: ReviewContext = agents[0].runs[0]
    contexts: list[ReviewContext] = [fc.runs[0], da.runs[0], ce.runs[0]]
    assert all(c.meter is extract_ctx.meter for c in contexts)
    assert all(c.ledger is contexts[0].ledger for c in contexts)
    assert len(contexts[0].ledger.entries) == 6
    assert extract_ctx.plan is None
    assert all(c.plan is not None and c.plan is contexts[0].plan for c in contexts)


async def test_default_registry_and_missing_setup() -> None:
    with pytest.raises(MissingSetup):
        await pipeline.run_review(OPINION_DOC, Profile.OPINION, ProgressRecorder())
    pipeline.set_default_registry(make_registry(standard_agents()))
    report = await pipeline.run_review(OPINION_DOC, Profile.OPINION, ProgressRecorder())
    assert report.document_id == OPINION_DOC.id


@pytest.mark.parametrize("doc", sample_documents(), ids=lambda d: d.id)
async def test_every_sample_document_runs_on_empty_fakes(doc: Document) -> None:
    agents = [FakeAgent(n.value) for n in AgentName if n is not AgentName.ORCHESTRATOR]
    rec = ProgressRecorder()
    report = await pipeline.run_review(doc, Profile.AUTO, rec, registry=make_registry(agents))
    assert report.document_id == doc.id
    assert report.verdict_line == "Ready: no issues found."
    assert_progress(rec)


async def test_progress_callback_errors_are_swallowed() -> None:
    def broken(_event: ProgressEvent) -> None:
        raise RuntimeError("client went away")

    report = await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, broken, registry=make_registry(standard_agents())
    )
    assert report.must_fix


# ---------------------------------------------------------------------------
# Classification and planning
# ---------------------------------------------------------------------------


async def test_auto_classification_with_llm() -> None:
    llm = FakeLLM({"orchestrator.classify": {"profile": "design_doc"}})
    agents = standard_agents()
    report = await pipeline.run_review(
        OPINION_DOC, Profile.AUTO, ProgressRecorder(), registry=make_registry(agents, llm)
    )
    assert report.profile is Profile.DESIGN_DOC
    structure, originality = agents[5], agents[4]
    assert len(structure.runs) == 1
    assert originality.runs == []
    assert [f.id for f in report.should_fix] == ["find_structure"]
    assert llm.calls[0].tag == "orchestrator.classify"


@pytest.mark.parametrize(
    ("doc", "expected"), [(DESIGN_DOC, Profile.DESIGN_DOC), (OPINION_DOC, Profile.OPINION)]
)
async def test_auto_classification_fallback(doc: Document, expected: Profile) -> None:
    llm = FakeLLM({"orchestrator.classify": ProviderError("provider down")})
    agents = [FakeAgent("extractor")]
    rec = ProgressRecorder()
    report = await pipeline.run_review(doc, Profile.AUTO, rec, registry=make_registry(agents, llm))
    assert report.profile is expected
    assert any("guessed" in e.message for e in rec.events)


async def test_valid_llm_plan_is_used() -> None:
    llm = FakeLLM(
        {
            "orchestrator.plan": {
                "agents": ["factcheck", "astrologer"],
                "deep_research_claim_ids": ["claim_goals"],
                "claim_budgets": [
                    {
                        "claim_id": "claim_goals",
                        "max_iterations": 3,
                        "max_search_calls": 5,
                        "max_tokens": 10_000,
                    }
                ],
                "rationale": "numbers only",
            }
        }
    )
    agents = standard_agents()
    await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, ProgressRecorder(), registry=make_registry(agents, llm)
    )
    fc, da, ce = agents[1], agents[2], agents[3]
    assert len(fc.runs) == 1
    assert da.runs == [] and ce.runs == []
    plan = fc.runs[0].plan
    assert plan is not None
    # Originality is required for opinion pieces even if the planner omits it.
    assert plan.agents == [AgentName.FACTCHECK, AgentName.ORIGINALITY]
    assert plan.deep_research_claim_ids == ["claim_goals"]
    assert plan.budget_for("claim_goals").max_search_calls == 5
    [call] = llm.calls_for("orchestrator.plan")
    assert call.model_tier == "strong"


@pytest.mark.parametrize(
    "bad_plan",
    [
        {
            "agents": ["factcheck"],
            "claim_budgets": [
                {
                    "claim_id": "claim_goals",
                    "max_iterations": 7,
                    "max_search_calls": 1,
                    "max_tokens": 1,
                }
            ],
        },
        {
            "agents": ["factcheck"],
            "claim_budgets": [
                {
                    "claim_id": "claim_goals",
                    "max_iterations": 3,
                    "max_search_calls": 999,
                    "max_tokens": 1,
                }
            ],
        },
        {"agents": ["astrologer"]},
        "not json at all",
        ProviderError("planner down"),
    ],
    ids=["iterations", "over-budget", "no-agents", "garbage", "error"],
)
async def test_invalid_plan_falls_back_to_profile_default(bad_plan: Any) -> None:
    llm = FakeLLM({"orchestrator.plan": bad_plan})
    agents = standard_agents()
    await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, ProgressRecorder(), registry=make_registry(agents, llm)
    )
    fc, da, ce, orig, structure = agents[1:6]
    assert all(len(a.runs) == 1 for a in (fc, da, ce, orig))
    assert structure.runs == []
    plan = fc.runs[0].plan
    assert plan is not None
    assert plan.rationale.startswith("profile default")
    assert plan.agents == [FC, DA, CE, AgentName.ORIGINALITY]


async def test_missing_agents_are_not_scheduled_and_noted() -> None:
    agents = [extractor(), standard_factchecker()]
    report = await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, ProgressRecorder(), registry=make_registry(agents)
    )
    plan = agents[1].runs[0].plan
    assert plan is not None
    assert plan.agents == [FC]
    assert "devil's advocate not configured" in report.notes
    assert "copy edit not configured" in report.notes


# ---------------------------------------------------------------------------
# Degradation and time limits
# ---------------------------------------------------------------------------


async def test_raising_agent_degrades_report() -> None:
    agents = standard_agents()
    agents[1] = FakeFactChecker(raises=RuntimeError("boom with secret-ish detail"))
    agents[3] = FakeAgent("copyedit", raises=ProviderError("down"))
    rec = ProgressRecorder()
    report = await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, rec, registry=make_registry(agents)
    )
    assert "fact-check unavailable" in report.notes
    assert "copy edit unavailable" in report.notes
    assert not any("secret-ish" in n for n in report.notes)
    assert report.must_fix == []
    assert [r.id for r in report.counter_case] == ["reb_fatigue", "reb_shots", "reb_extra"]
    assert report.verdict_line.endswith("Partial review: see notes.")
    assert agents[1].rechecks == []
    assert_progress(rec)


async def test_agent_errors_and_notes_are_copied() -> None:
    agents = standard_agents()
    agents[2] = FakeDevilsAdvocate(error="search budget exhausted")
    agents[4] = FakeAgent(
        "originality",
        on_run=lambda ctx: AgentResult(agent="originality", notes=["sampled 3 sentences"]),
    )
    report = await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, ProgressRecorder(), registry=make_registry(agents)
    )
    assert "devil's advocate: search budget exhausted" in report.notes
    assert "sampled 3 sentences" in report.notes


async def test_unchecked_claims_add_partial_fact_check_note() -> None:
    agents = standard_agents()
    agents[1] = FakeFactChecker(
        ledger_updates=[
            SetVerdict(claim_id="claim_points", verdict=Verdict.UNCHECKED, note="budget exhausted"),
        ]
    )
    report = await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, ProgressRecorder(), registry=make_registry(agents)
    )
    assert "partial fact-check: 3 claim(s) unchecked" in report.notes


async def test_invalid_ledger_updates_are_skipped() -> None:
    bad = rebuttal("reb_ghost", ["claim_missing"], evidence=[ev()])
    agents = standard_agents()
    agents[2] = FakeDevilsAdvocate(
        ledger_updates=[AddRebuttal(rebuttal=bad)],
        findings=[finding("find_reb_ghost", DA, Severity.STRONG_REBUTTAL)],
    )
    report = await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, ProgressRecorder(), registry=make_registry(agents)
    )
    assert "find_reb_ghost" not in report.ledger.findings
    assert "devil's advocate: 1 invalid ledger update(s) skipped" in report.notes


async def test_timeout_returns_partial_results() -> None:
    agents = standard_agents()
    agents[3] = FakeAgent("copyedit", delay=30, findings=[finding("slow", CE, Severity.STYLE)])
    rec = ProgressRecorder()
    started = time.monotonic()
    report = await pipeline.run_review(
        OPINION_DOC,
        Profile.OPINION,
        rec,
        registry=make_registry(agents),
        budget=Budget(max_seconds=0.5),
    )
    assert time.monotonic() - started < 5
    assert "partial results: time limit reached (copy edit did not finish)" in report.notes
    assert [f.id for f in report.must_fix] == ["find_goals", "find_fatigue"]
    assert "slow" not in report.ledger.findings
    assert report.verdict_line.endswith("Partial review: see notes.")
    assert_progress(rec)


async def test_timeout_during_extraction() -> None:
    agents = [
        FakeAgent("extractor", delay=30, ledger_updates=claim_updates()),
        standard_factchecker(),
    ]
    report = await pipeline.run_review(
        OPINION_DOC,
        Profile.OPINION,
        ProgressRecorder(),
        registry=make_registry(agents),
        budget=Budget(max_seconds=0.2),
    )
    assert report.ledger.entries == {}
    assert agents[1].runs == []
    assert any(
        n.startswith("partial results: time limit reached (claim extraction") for n in report.notes
    )


# ---------------------------------------------------------------------------
# Reaction round
# ---------------------------------------------------------------------------

FACT_ATTACK_EVIDENCE = ev("https://ex.org/points-wrong", primary=True)


def fact_attack(rid: str, cid: str, *, evidence: bool = True, strength: int = 4) -> Any:
    return rebuttal(
        rid,
        [cid],
        target=RebuttalTarget.FACT,
        evidence=[FACT_ATTACK_EVIDENCE] if evidence else [],
        strength=strength,
    )


def reaction_agents(
    rebuttals: list[Any], fc: FakeFactChecker, da_kwargs: dict[str, Any] | None = None
) -> list[Any]:
    ledger = sample_ledger()
    da = FakeDevilsAdvocate(
        ledger_updates=[AddRebuttal(rebuttal=r) for r in rebuttals],
        findings=[da_finding_for(r, ledger) for r in rebuttals],
        **(da_kwargs or {}),
    )
    return [extractor(), fc, da]


async def test_recheck_once_per_claim_and_verdict_update() -> None:
    fc = FakeFactChecker(
        ledger_updates=verdicts(claim_points=Verdict.VERIFIED),
        recheck_result=AgentResult(
            agent="factcheck",
            ledger_updates=[
                SetVerdict(
                    claim_id="claim_points", verdict=Verdict.WRONG, evidence=[FACT_ATTACK_EVIDENCE]
                )
            ],
            findings=[
                finding(
                    "find_points_wrong",
                    FC,
                    Severity.FACTUAL_ERROR,
                    claim_span("claim_points"),
                    claim_ids=["claim_points"],
                )
            ],
        ),
    )
    rebs = [
        fact_attack("reb_a", "claim_points"),
        fact_attack("reb_b", "claim_points"),
        rebuttal("reb_interp", ["claim_points"], evidence=[ev()]),  # interpretation: no re-check
        fact_attack("reb_nosrc", "claim_points", evidence=False),  # no evidence: no re-check
    ]
    agents = reaction_agents(rebs, fc)
    report = await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, ProgressRecorder(), registry=make_registry(agents)
    )

    assert [cid for cid, _ in fc.rechecks] == ["claim_points"]
    assert [e.url for e in fc.rechecks[0][1]] == [FACT_ATTACK_EVIDENCE.url]
    entry = report.ledger.get("claim_points")
    assert entry.rechecked is True
    assert entry.verdict is Verdict.WRONG
    assert fc.debates == []  # no longer in dispute after the re-check
    must_fix = report.must_fix
    assert [f.id for f in must_fix] == ["find_points_wrong"]
    assert set(must_fix[0].merged_from) >= {FC, DA}  # rule 3 merged the rebuttal findings in
    surviving = {r.id for r in entry.rebuttals}
    assert {"reb_a", "reb_b", "reb_interp"} <= surviving


async def test_no_recheck_without_rechecker() -> None:
    fc = FakeAgent("factcheck", ledger_updates=verdicts(claim_points=Verdict.VERIFIED))
    agents = [
        extractor(),
        fc,
        reaction_agents([fact_attack("reb_a", "claim_points")], FakeFactChecker())[2],
    ]
    report = await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, ProgressRecorder(), registry=make_registry(agents)
    )
    entry = report.ledger.get("claim_points")
    assert entry.rechecked is False
    assert entry.verdict is Verdict.VERIFIED
    assert entry.rebuttals == []  # rule 2 discarded the fact attack
    assert "find_reb_a" not in report.ledger.findings


async def test_debate_top_three_only_with_code_fallback_ruling() -> None:
    disputed = ["claim_goals", "claim_shots", "claim_points", "claim_fatigue"]  # 0.8,0.7,0.6,0.5
    fc = FakeFactChecker(
        ledger_updates=verdicts(**dict.fromkeys(disputed, Verdict.VERIFIED)),
    )
    rebs = [fact_attack(f"reb_{cid}", cid) for cid in disputed]
    agents = reaction_agents(rebs, fc)
    llm = FakeLLM({"orchestrator.debate_ruling": ProviderError("down")})
    report = await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, ProgressRecorder(), registry=make_registry(agents, llm)
    )
    da = agents[2]
    assert [cid for cid, _ in fc.rechecks] == disputed  # every disputed claim re-checked once
    assert [cid for cid, _ in fc.debates] == ["claim_goals", "claim_shots", "claim_points"]
    assert [cid for cid, _ in da.debates] == ["claim_goals", "claim_shots", "claim_points"]
    assert [e.url for e in fc.debates[0][1]] == [FACT_ATTACK_EVIDENCE.url]
    assert [e.url for e in da.debates[0][1]] == ["https://example.org/claim_goals"]
    for cid in disputed[:3]:
        record = report.ledger.get(cid).debate
        assert record is not None
        assert record.final_verdict is Verdict.VERIFIED
        assert record.reasoning.startswith("Code ruling")
        assert report.ledger.get(cid).rebuttals == []  # rule 2 after the verified ruling
    assert report.ledger.get("claim_fatigue").debate is None
    assert len(llm.calls_for("orchestrator.debate_ruling")) == 3


async def test_debate_llm_ruling_overturns_verdict() -> None:
    fc = FakeFactChecker(
        ledger_updates=verdicts(claim_points=Verdict.VERIFIED),
        debate_position=DebatePosition(
            agent="factcheck", claim_id="claim_points", argument="Table says 99."
        ),
    )
    agents = reaction_agents([fact_attack("reb_a", "claim_points")], fc)
    llm = FakeLLM(
        {
            "orchestrator.debate_ruling": {
                "winner": "devils_advocate",
                "final_verdict": "wrong",
                "ruling": "The primary source contradicts the claim.",
                "reasoning": "Counter-evidence is primary; the cited table is secondary.",
            }
        }
    )
    report = await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, ProgressRecorder(), registry=make_registry(agents, llm)
    )
    entry = report.ledger.get("claim_points")
    assert entry.debate is not None
    assert entry.debate.factcheck_position == "Table says 99."
    assert entry.debate.devils_advocate_position == "The rebuttal stands."
    assert entry.debate.final_verdict is Verdict.WRONG
    assert entry.verdict is Verdict.WRONG
    assert [f.id for f in report.must_fix] == ["find_debate_claim_points"]
    assert report.must_fix[0].severity is Severity.FACTUAL_ERROR
    assert [r.id for r in entry.rebuttals] == ["reb_a"]  # survives: claim no longer verified
    [call] = llm.calls_for("orchestrator.debate_ruling")
    assert call.model_tier == "strong"


async def test_inconsistent_llm_ruling_uses_code_fallback() -> None:
    fc = FakeFactChecker(
        ledger_updates=verdicts(claim_points=Verdict.VERIFIED),
        debate_position=DebatePosition(
            agent="factcheck", claim_id="claim_points", argument="I concede.", concedes=True
        ),
    )
    agents = reaction_agents([fact_attack("reb_a", "claim_points")], fc)
    llm = FakeLLM(
        {
            "orchestrator.debate_ruling": {
                "winner": "factcheck",
                "final_verdict": "wrong",
                "ruling": "x",
                "reasoning": "y",
            }
        }
    )
    report = await pipeline.run_review(
        OPINION_DOC, Profile.OPINION, ProgressRecorder(), registry=make_registry(agents, llm)
    )
    entry = report.ledger.get("claim_points")
    assert entry.debate is not None and entry.debate.reasoning.startswith("Code ruling")
    assert entry.verdict is Verdict.WRONG  # fact-checker conceded


async def test_orchestrator_class_direct_use() -> None:
    orch = Orchestrator(make_registry(standard_agents()), budget=Budget(max_seconds=30))
    report = await orch.review(OPINION_DOC, Profile.OPINION)
    assert report.document_id == OPINION_DOC.id
    assert report.must_fix
