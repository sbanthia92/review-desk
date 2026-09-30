"""ProgressTracker rescaling and profile classification."""

from __future__ import annotations

from reviewdesk.contracts import Profile, ProgressEvent, ProgressStep, ProviderError
from reviewdesk.orchestrator.classify import classify, heuristic_profile
from reviewdesk.orchestrator.progress import STEP_RANGES, ProgressTracker
from reviewdesk.testing.fakes import (
    DESIGN_DOC,
    ESSAY_DOC,
    OPINION_DOC,
    REPORT_DOC,
    SHORT_DOC,
    FakeLLM,
    ProgressRecorder,
)


def test_tracker_rescales_parallel_agents_into_step_range() -> None:
    rec = ProgressRecorder()
    tracker = ProgressTracker(rec)
    tracker.start_parts(ProgressStep.REVIEWING, ["a", "b"], "go")
    lo, hi = STEP_RANGES[ProgressStep.REVIEWING]
    tracker.callback_for(ProgressStep.REVIEWING, "a")(
        ProgressEvent(step="whatever", message="a half", percent=50)
    )
    assert rec.events[-1].percent == lo + (hi - lo) * 0.25
    assert rec.events[-1].step == "reviewing"
    assert rec.events[-1].message == "a half"
    tracker.callback_for(ProgressStep.REVIEWING, "a")(
        ProgressEvent(step="x", message="backwards", percent=10)
    )
    assert rec.events[-1].percent == lo + (hi - lo) * 0.25  # never goes back
    tracker.part(ProgressStep.REVIEWING, "b", 1.0, "b done")
    assert rec.events[-1].percent == lo + (hi - lo) * 0.75


def test_tracker_drops_events_for_earlier_steps_and_swallows_errors() -> None:
    calls: list[ProgressEvent] = []

    def broken(event: ProgressEvent) -> None:
        calls.append(event)
        raise RuntimeError("sink down")

    tracker = ProgressTracker(broken)
    tracker.emit(ProgressStep.PLANNING, "plan", 1.0)
    late = tracker.callback_for(ProgressStep.EXTRACTING, "extractor")
    late(ProgressEvent(step="extracting", message="late", percent=90))
    tracker.emit(ProgressStep.EXTRACTING, "late too", 1.0)
    assert [e.message for e in calls] == ["plan"]
    tracker.emit(ProgressStep.DONE, "done", 1.0)
    assert tracker.percent == 100.0


async def test_classify_uses_llm_when_auto() -> None:
    llm = FakeLLM({"orchestrator.classify": {"profile": "design_doc", "reason": "RFC"}})
    result = await classify(OPINION_DOC, Profile.AUTO, llm)
    assert result.profile is Profile.DESIGN_DOC
    assert result.method == "llm"
    [call] = llm.calls
    assert call.tag == "orchestrator.classify"
    assert call.model_tier == "cheap"
    assert "<document>" in call.messages[-1].content
    assert call.messages[0].role == "system"
    assert OPINION_DOC.text[:40] not in call.messages[0].content  # document never in system


async def test_classify_skips_llm_for_explicit_profile() -> None:
    llm = FakeLLM()
    result = await classify(OPINION_DOC, Profile.DESIGN_DOC, llm)
    assert result.profile is Profile.DESIGN_DOC
    assert result.method == "user"
    assert llm.calls == []


async def test_classify_falls_back_to_heuristic() -> None:
    for reply in (ProviderError("down"), {"profile": "poetry"}, "not json"):
        llm = FakeLLM({"orchestrator.classify": reply})
        design = await classify(DESIGN_DOC, Profile.AUTO, llm)
        opinion = await classify(OPINION_DOC, Profile.AUTO, llm)
        assert (design.profile, design.method) == (Profile.DESIGN_DOC, "heuristic")
        assert (opinion.profile, opinion.method) == (Profile.OPINION, "heuristic")


def test_heuristic_profile_on_samples() -> None:
    assert heuristic_profile(DESIGN_DOC) is Profile.DESIGN_DOC
    for doc in (OPINION_DOC, REPORT_DOC, ESSAY_DOC, SHORT_DOC):
        assert heuristic_profile(doc) is Profile.OPINION
