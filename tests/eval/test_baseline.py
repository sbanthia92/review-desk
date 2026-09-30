"""The single-prompt baseline: prompt shape and conversion to a Report."""

from __future__ import annotations

from eval.baseline import (
    TAG_BASELINE,
    BaselineIssue,
    BaselineReview,
    BaselineReviewer,
    BaselineSource,
    review_to_report,
)
from eval.items import report_items
from eval.metrics import score_report
from reviewdesk.contracts import ModelTier, Profile, RebuttalTarget, Severity
from reviewdesk.testing.fakes import FakeLLM, ProgressRecorder
from tests.eval.helpers import TEXT, span, tiny_seeded

REVIEW = BaselineReview(
    verdict_line="Needs work.",
    issues=[
        BaselineIssue(
            kind="factual_error",
            quote="Liverpool won the 2019-20 league with 105 points.",
            problem="They had 99 points.",
            sources=[BaselineSource(url="https://example.org/t", title="Table", quote="99 points")],
        ),
        BaselineIssue(
            kind="unsupported",
            quote="pressing teams concede 50%  fewer shots",
            problem="No source.",
        ),
        BaselineIssue(
            kind="counterargument",
            quote="If one pressing team won, pressing always wins.",
            problem="One title is an outlier.",
            strength=4,
        ),
        BaselineIssue(
            kind="counterargument",
            quote="If one pressing team won, pressing always wins.",
            problem="Selection bias.",
            strength=2,
        ),
        BaselineIssue(kind="counterargument", quote="Not in the text", problem="Floating."),
        BaselineIssue(kind="style", quote="The players was tired.", problem="was -> were"),
        BaselineIssue(kind="structure", quote="", problem="No conclusion."),
        BaselineIssue(
            kind="originality",
            quote="It is not from the benevolence of the butcher that we expect our dinner.",
            problem="Adam Smith.",
        ),
    ],
)


def test_review_to_report_sections_spans_and_rebuttals() -> None:
    doc = tiny_seeded().document
    report = review_to_report(doc, Profile.OPINION, REVIEW)
    assert [f.severity for f in report.must_fix] == [Severity.FACTUAL_ERROR, Severity.UNSUPPORTED]
    assert report.must_fix[0].span == span("Liverpool won the 2019-20 league with 105 points.")
    assert report.must_fix[1].span is not None  # located despite case/whitespace changes
    assert report.must_fix[0].evidence[0].excerpt == "99 points"
    # Two counterarguments on one quote share one claim; unlocatable one is document-level.
    assert len(report.ledger.entries) == 1
    assert [r.strength for r in report.counter_case] == [4, 2]
    assert all(r.target is RebuttalTarget.INTERPRETATION for r in report.counter_case)
    floating = [f for f in report.should_fix if f.span is None and f.severity is Severity.CONSIDER]
    assert [f.message for f in floating] == ["Floating."]
    assert report.polish[0].severity is Severity.STYLE
    assert report.originality[0].heuristic
    assert report.notes == ["1 quoted issue(s) could not be located"]
    for f in (*report.must_fix, *report.polish, *report.originality):
        assert f.span is None or f.span.text_of(TEXT)


async def test_baseline_reviewer_prompt_and_scoring() -> None:
    llm = FakeLLM({TAG_BASELINE: REVIEW})
    progress = ProgressRecorder()
    seeded = tiny_seeded()
    report = await BaselineReviewer(llm)(seeded.document, Profile.AUTO, progress)
    call = llm.calls[0]
    assert call.tag == TAG_BASELINE and call.model_tier is ModelTier.STRONG
    assert call.schema is BaselineReview
    system, user = call.messages
    assert "argue the other side" in system.content and TEXT not in system.content
    assert user.content.startswith("<document>") and TEXT.strip() in user.content
    assert report.profile is Profile.OPINION  # AUTO treated as OPINION
    assert report.usage.llm_calls == 1 and report.usage.input_tokens > 0
    assert progress.steps[-1] == "done"

    score = await score_report(seeded, report, pipeline="baseline")
    caught = {o.defect_id for o in score.defects if o.caught}
    assert caught == {"wrong", "unsup", "weak", "gram", "borrowed"}
    assert score.items == len(report_items(report))


async def test_design_doc_prompt_variant() -> None:
    llm = FakeLLM({TAG_BASELINE: BaselineReview()})
    report = await BaselineReviewer(llm)(tiny_seeded().document, Profile.DESIGN_DOC, lambda e: None)
    assert "against the design" in llm.calls[0].messages[0].content
    assert report.verdict_line and report.must_fix == []
