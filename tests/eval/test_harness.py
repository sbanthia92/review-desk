"""The harness end to end: FakeAgent pipelines, the baseline on FakeLLM, the CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from eval.baseline import TAG_BASELINE, BaselineReviewer
from eval.cli import main, resolve_pipelines
from eval.dataset import RECALL_TYPES, DefectType, SeededDocument, load_dataset
from eval.fake_pipeline import (
    FakeAgentPipeline,
    OracleAgents,
    assemble_report,
    builtin_pipelines,
    oracle_agents,
)
from eval.harness import run_eval, run_one, save_results
from eval.judge import TAG_CITATION, TAG_REAL, TAG_REBUTTAL, LLMJudge
from eval.metrics import ConflictStatus
from eval.results import Stat, summarize
from reviewdesk.contracts import (
    AddClaim,
    AgentName,
    Budget,
    Claim,
    ClaimLedger,
    ClaimType,
    Document,
    Finding,
    Message,
    Profile,
    ProgressCallback,
    ProgressEvent,
    Report,
    ReviewContext,
    ReviewPipeline,
    Severity,
    Span,
    Usage,
)
from reviewdesk.testing.fakes import (
    FakeAgent,
    FakeFetcher,
    FakeLLM,
    FakeSearch,
    ProgressRecorder,
)

DATASET = load_dataset()


# -- a hand-written fake run_review (stand-in for the T10 orchestrator) -----------


def fake_run_review_for(dataset: list[SeededDocument]) -> ReviewPipeline:
    """A ``run_review``-shaped callable built from ``FakeAgent`` outputs.

    It catches wrong facts (fact-checker) and grammar (copy editor) only, so
    recall differs by defect type.
    """
    by_id = {d.id: d for d in dataset}

    async def run_review(doc: Document, profile: Profile, on_progress: ProgressCallback) -> Report:
        seeded = by_id[doc.id]
        on_progress(ProgressEvent(step="reviewing", message="go", percent=10))
        agents = [
            FakeAgent(
                AgentName.FACTCHECK,
                findings=[
                    Finding(
                        agent=AgentName.FACTCHECK,
                        severity=Severity.FACTUAL_ERROR,
                        span=d.located,
                        message="Wrong.",
                    )
                    for d in seeded.of_type(DefectType.WRONG_FACT)
                ],
                usage=Usage(input_tokens=1000, output_tokens=200, search_calls=4, llm_calls=3),
            ),
            FakeAgent(
                AgentName.COPYEDIT,
                findings=[
                    Finding(
                        agent=AgentName.COPYEDIT,
                        severity=Severity.STYLE,
                        span=d.located,
                        message="Grammar.",
                    )
                    for d in seeded.of_type(DefectType.GRAMMAR)
                ],
                usage=Usage(input_tokens=300, output_tokens=50, llm_calls=1),
            ),
        ]
        ledger = ClaimLedger()
        ctx = ReviewContext(
            document=doc,
            profile=profile,
            ledger=ledger,
            llm=FakeLLM(),
            search=FakeSearch(),
            fetcher=FakeFetcher(),
            budget=Budget(),
            emit_progress=on_progress,
        )
        results = await asyncio.gather(*(a.run(ctx) for a in agents))
        usage = Usage()
        for r in results:
            ledger.apply_result(r)
            usage = usage + r.usage
        return Report(
            document_id=doc.id,
            profile=profile,
            verdict_line="fake",
            must_fix=[f for r in results for f in r.findings if f.severity >= Severity.UNSUPPORTED],
            polish=[f for r in results for f in r.findings if f.severity is Severity.STYLE],
            ledger=ledger,
            usage=usage,
        )

    return run_review


async def test_harness_end_to_end_with_fake_run_review() -> None:
    pipeline = fake_run_review_for(DATASET)
    results = await run_eval({"fake": pipeline}, DATASET, repeats=2)
    assert len(results.scores) == 2 * len(DATASET)
    summary = results.summary()["fake"]
    assert summary["recall.wrong_fact"].mean == 1.0
    assert summary["recall.grammar"].mean is not None and summary["recall.grammar"].mean > 0.9
    assert summary["recall.weak_argument"].mean == 0.0
    assert summary["recall.borrowed_sentence"].mean == 0.0
    assert summary["precision_matched"].mean == 1.0
    assert summary["tokens_per_doc"].mean == 1550.0
    assert summary["search_calls_per_doc"].mean == 4.0
    assert summary["precision"].mean is None  # no judge
    assert summary["errors"].mean == 0.0
    table = results.table()
    assert "| Metric" in table and "fake" in table and "Recall: wrong_fact" in table


async def test_oracle_pipelines_and_conflict_rules() -> None:
    pipes = builtin_pipelines(DATASET)
    results = await run_eval(pipes, DATASET, repeats=3)
    s = results.summary()
    assert s["oracle"]["recall.overall"].mean == 1.0
    assert s["oracle"]["precision_matched"].mean == 1.0
    assert s["oracle"]["conflict_accuracy"].mean == 1.0
    assert s["oracle"]["conflicts_triggered"].mean == 1.0
    assert s["oracle-no-rules"]["conflict_accuracy"].mean == 0.0
    assert s["empty"]["recall.overall"].mean == 0.0
    noisy = s["noisy"]["recall.overall"]
    assert noisy.mean is not None and 0.3 < noisy.mean < 1.0
    assert noisy.stdev is not None and noisy.stdev > 0  # repeats vary
    table = results.table()
    for name in pipes:
        assert name in table
    assert "±" in table


async def test_every_conflict_rule_resolved_by_oracle() -> None:
    results = await run_eval({"oracle": builtin_pipelines(DATASET)["oracle"]}, DATASET)
    statuses = {(c.rule, c.status) for s in results.scores for c in s.conflicts}
    assert all(status is ConflictStatus.RESOLVED for _, status in statuses)
    assert len({rule for rule, _ in statuses}) == 4


async def test_oracle_is_perfect_on_every_one_of_the_thirty_documents() -> None:
    assert len(DATASET) >= 30
    results = await run_eval({"oracle": builtin_pipelines(DATASET)["oracle"]}, DATASET)
    assert {s.doc_id for s in results.scores} == {d.id for d in DATASET}
    for score in results.scores:
        assert score.error is None, score.doc_id
        assert all(o.caught for o in score.defects), score.doc_id
        assert score.items == score.items_matched, score.doc_id  # no stray findings
        assert score.conflicts, score.doc_id
        for c in score.conflicts:
            assert c.status is ConflictStatus.RESOLVED, (score.doc_id, c.defect_id, c.detail)
    summary = results.summary()["oracle"]
    for defect_type in RECALL_TYPES:
        assert summary[f"recall.{defect_type.value}"].mean == 1.0
    assert summary["conflict_accuracy"].mean == 1.0


async def test_empty_and_noisy_pipelines_on_the_full_set() -> None:
    pipes = builtin_pipelines(DATASET)
    results = await run_eval({"noisy": pipes["noisy"], "empty": pipes["empty"]}, DATASET)
    s = results.summary()
    assert s["empty"]["recall.overall"].mean == 0.0
    assert s["empty"]["conflicts_triggered"].mean == 0.0
    noisy = s["noisy"]["recall.overall"].mean
    assert noisy is not None and 0.5 < noisy < 0.9
    precision = s["noisy"]["precision_matched"].mean
    assert precision is not None and precision < 1.0  # one false positive per document


async def test_baseline_through_harness_with_fake_llm_and_judge() -> None:
    by_text = {d.text.strip(): d for d in DATASET}

    def reply(messages: list[Message], schema: object) -> dict[str, Any]:
        user = messages[-1].content
        seeded = next(d for t, d in by_text.items() if t in user)
        issues: list[dict[str, Any]] = [
            {"kind": "factual_error", "quote": d.quote, "problem": "wrong"}
            for d in seeded.of_type(DefectType.WRONG_FACT)
        ]
        issues += [
            {"kind": "counterargument", "quote": d.quote, "problem": "rebut", "strength": 3}
            for d in seeded.of_type(DefectType.WEAK_ARGUMENT)
        ]
        issues.append({"kind": "style", "quote": "", "problem": "generic advice"})
        return {"verdict_line": "meh", "issues": issues}

    llm = FakeLLM(
        {
            TAG_BASELINE: reply,
            TAG_REAL: {"real": False},
            TAG_CITATION: {"supports": True},
            TAG_REBUTTAL: {"score": 3},
        }
    )
    results = await run_eval(
        {"baseline": BaselineReviewer(llm)}, DATASET[:3], judge=LLMJudge(llm), repeats=1
    )
    s = results.summary()["baseline"]
    assert s["recall.wrong_fact"].mean == 1.0 and s["recall.weak_argument"].mean == 1.0
    assert s["recall.grammar"].mean == 0.0
    assert s["rebuttal_strength"].mean == 3.0
    assert s["precision"].mean is not None and s["precision"].mean < 1.0
    assert s["llm_calls_per_doc"].mean == 1.0
    assert len(llm.calls_for(TAG_BASELINE)) == 3
    assert llm.calls_for(TAG_REBUTTAL) and llm.calls_for(TAG_REAL)


async def test_failures_and_timeouts_are_scored_not_raised() -> None:
    seeded = DATASET[0]

    async def boom(doc: Document, profile: Profile, on_progress: ProgressCallback) -> Report:
        raise RuntimeError("provider exploded\nsecond line")

    async def slow(doc: Document, profile: Profile, on_progress: ProgressCallback) -> Report:
        await asyncio.sleep(5)
        raise AssertionError("unreachable")

    failed = await run_one("boom", boom, seeded)
    assert failed.error == "RuntimeError: provider exploded"
    assert not any(o.caught for o in failed.defects)
    timed = await run_one("slow", slow, seeded, time_limit=0.01)
    assert timed.error == "timeout"
    results = await run_eval({"boom": boom}, DATASET[:2], repeats=2)
    assert results.summary()["boom"]["errors"].mean == 2.0


async def test_auto_profile_is_passed_through() -> None:
    seen: list[Profile] = []

    async def spy(doc: Document, profile: Profile, on_progress: ProgressCallback) -> Report:
        seen.append(profile)
        return Report(document_id=doc.id, profile=Profile.OPINION, verdict_line="x")

    await run_eval({"spy": spy}, DATASET[:1], auto_profile=True)
    await run_eval({"spy": spy}, DATASET[:1])
    assert seen == [Profile.AUTO, DATASET[0].profile]


async def test_fake_agent_pipeline_degrades_on_raising_agent() -> None:
    def agents(doc: Document, profile: Profile) -> list[FakeAgent]:
        claim = Claim(
            id="c1",
            text=doc.text[:5],
            span=Span(start=0, end=5),
            type=ClaimType.FACTUAL,
            importance=0.5,
        )
        return [
            FakeAgent(AgentName.EXTRACTOR, ledger_updates=[AddClaim(claim=claim)]),
            FakeAgent(AgentName.FACTCHECK, raises=RuntimeError("down")),
            FakeAgent(AgentName.COPYEDIT, error="rate limited"),
        ]

    recorder = ProgressRecorder()
    report = await FakeAgentPipeline(agents)(DATASET[0].document, Profile.AUTO, recorder)
    assert "c1" in report.ledger.entries
    assert any("factcheck unavailable" in n for n in report.notes)
    assert any("copyedit unavailable" in n for n in report.notes)
    assert recorder.steps[-1] == "done" and report.profile is Profile.OPINION


def test_oracle_agents_follow_profile_and_assembly_rules() -> None:
    design = next(d for d in DATASET if d.profile is Profile.DESIGN_DOC)
    names = [a.name for a in oracle_agents(design)]
    assert AgentName.ORIGINALITY not in names and AgentName.STRUCTURE in names
    opinion = next(d for d in DATASET if d.profile is Profile.OPINION)
    names = [a.name for a in oracle_agents(opinion)]
    assert AgentName.STRUCTURE not in names and AgentName.ORIGINALITY in names
    assert names[0] == AgentName.EXTRACTOR

    ledger = ClaimLedger()
    report = assemble_report(design.document, Profile.DESIGN_DOC, ledger)
    assert report.must_fix == [] and report.counter_case == []


def test_oracle_factory_is_reproducible_and_varies_per_call() -> None:
    doc = DATASET[0].document
    a = OracleAgents(DATASET, miss_rate=0.5, seed=1)
    b = OracleAgents(DATASET, miss_rate=0.5, seed=1)

    def counts(factory: OracleAgents) -> list[int]:
        return [sum(len(ag.findings) for ag in factory(doc, Profile.OPINION)) for _ in range(4)]

    first, second = counts(a), counts(b)
    assert first == second
    assert len(set(first)) > 1
    assert OracleAgents(DATASET)(Document.from_text("x", id="unknown"), Profile.OPINION) == []


def test_summarize() -> None:
    assert summarize([None, None]) == Stat(mean=None, stdev=None, n=0)
    assert summarize([0.5]) == Stat(mean=0.5, stdev=None, n=1)
    stat = summarize([0.4, 0.6, None])
    assert (
        stat.n == 2
        and stat.mean == pytest.approx(0.5)
        and stat.stdev == pytest.approx(0.1414, 1e-3)
    )


async def test_run_eval_validates_arguments() -> None:
    with pytest.raises(ValueError, match="repeats"):
        await run_eval({"x": builtin_pipelines(DATASET)["empty"]}, DATASET, repeats=0)
    with pytest.raises(ValueError, match="no pipelines"):
        await run_eval({}, DATASET)


async def test_save_results(tmp_path: Path) -> None:
    results = await run_eval({"oracle": builtin_pipelines(DATASET)["oracle"]}, DATASET[:2])
    json_path, md_path = save_results(results, tmp_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["pipelines"] == ["oracle"] and len(data["scores"]) == 2
    assert "| Metric" in md_path.read_text(encoding="utf-8")


# -- CLI ----------------------------------------------------------------------------


def test_cli_run_prints_results_table(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code = main(
        ["run", "-p", "oracle", "-p", "mine=noisy", "--repeats", "2", "--out", str(tmp_path)]
    )
    out, err = capsys.readouterr()
    assert code == 0
    assert "| Metric" in out and "oracle" in out and "mine" in out
    assert "Recall: overall" in out and "Conflict-resolution accuracy" in out
    assert "caught" in err  # per-document progress on stderr
    assert list(tmp_path.glob("*.json")) and list(tmp_path.glob("*.md"))


def test_cli_validate_list_and_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate"]) == 0
    assert "all spans valid" in capsys.readouterr().out
    assert main(["list"]) == 0
    assert "01-leicester-money" in capsys.readouterr().out
    assert main(["run", "-p", "baseline", "--no-save"]) == 2
    assert "needs --llm" in capsys.readouterr().err
    assert main(["run", "-p", "nope", "--no-save"]) == 2
    assert main(["run", "--judge", "--no-save"]) == 2
    assert main(["run", "--docs", "missing-doc", "--no-save"]) == 2
    assert main(["run", "-p", "no.such.module:x", "--no-save"]) == 2


def fake_llm_factory() -> FakeLLM:
    """Referenced by the CLI test via ``--llm tests.eval.test_harness:fake_llm_factory``."""
    return FakeLLM({TAG_BASELINE: {"verdict_line": "ok", "issues": []}}, default={"real": True})


def fake_pipeline_spec(doc: Document, profile: Profile, on_progress: ProgressCallback) -> Any:
    """Referenced by the CLI test as an import-spec pipeline."""
    return builtin_pipelines(DATASET)["empty"](doc, profile, on_progress)


def test_cli_import_specs_and_baseline(capsys: pytest.CaptureFixture[str]) -> None:
    mod = __name__
    code = main(
        [
            "run",
            "-p",
            "baseline",
            "-p",
            f"spec={mod}:fake_pipeline_spec",
            "--llm",
            f"{mod}:fake_llm_factory",
            "--docs",
            "01-leicester-money,07-postgres-queue",
            "--no-save",
            "-q",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and "baseline" in out and "spec" in out
    pipes = resolve_pipelines([f"{mod}:fake_pipeline_spec"], DATASET, None)
    assert list(pipes) == ["fake_pipeline_spec"]
