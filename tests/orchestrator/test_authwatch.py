import pytest

from reviewdesk.contracts import (
    AgentName,
    AgentResult,
    AuthError,
    Profile,
    ProviderError,
    ReviewContext,
)
from reviewdesk.orchestrator import AgentRegistry, Orchestrator
from reviewdesk.testing.fakes import (
    OPINION_DOC,
    FakeAgent,
    FakeFetcher,
    FakeLLM,
    FakeSearch,
    ProgressRecorder,
)


def _registry(llm: FakeLLM, search: FakeSearch, agents: list[FakeAgent]) -> AgentRegistry:
    return AgentRegistry.of(agents, llm=llm, search=search, fetcher=FakeFetcher())


async def test_rejected_llm_key_raises_instead_of_empty_report():
    llm = FakeLLM({"orchestrator": AuthError("anthropic rejected the key (401)")})
    registry = _registry(llm, FakeSearch(), [FakeAgent(AgentName.EXTRACTOR)])
    with pytest.raises(AuthError):
        await Orchestrator(registry).review(OPINION_DOC, Profile.AUTO, ProgressRecorder())


async def test_rejected_search_key_seen_by_an_agent_raises():
    async def searching(ctx: ReviewContext) -> AgentResult:
        try:
            await ctx.search.search("q")
        except ProviderError as exc:  # agents degrade as usual
            return AgentResult(agent=AgentName.FACTCHECK, error=str(exc))
        return AgentResult(agent=AgentName.FACTCHECK)

    class SearchingAgent(FakeAgent):
        async def run(self, ctx: ReviewContext) -> AgentResult:
            return await searching(ctx)

    registry = _registry(
        FakeLLM(default={}),
        FakeSearch(error=AuthError("brave rejected the key (401)")),
        [FakeAgent(AgentName.EXTRACTOR), SearchingAgent(AgentName.FACTCHECK)],
    )
    with pytest.raises(AuthError):
        await Orchestrator(registry).review(OPINION_DOC, Profile.OPINION, ProgressRecorder())


async def test_other_provider_errors_still_degrade():
    registry = _registry(
        FakeLLM({"orchestrator": ProviderError("overloaded", retryable=True)}),
        FakeSearch(),
        [FakeAgent(AgentName.EXTRACTOR)],
    )
    report = await Orchestrator(registry).review(OPINION_DOC, Profile.AUTO, ProgressRecorder())
    assert report.document_id == OPINION_DOC.id


async def test_degraded_review_is_never_called_ready():
    registry = _registry(
        FakeLLM(default={}),
        FakeSearch(),
        [
            FakeAgent(AgentName.EXTRACTOR),
            FakeAgent(AgentName.FACTCHECK, error="provider error"),
        ],
    )
    report = await Orchestrator(registry).review(OPINION_DOC, Profile.OPINION, ProgressRecorder())
    assert report.verdict_line.startswith("Incomplete review")
    assert "Ready" not in report.verdict_line


async def test_could_not_verify_findings_get_their_own_section():
    from reviewdesk.contracts import Finding, Severity, Span

    unverified = Finding(
        id="f_unverified",
        agent=AgentName.FACTCHECK,
        severity=Severity.CONSIDER,
        span=Span(start=0, end=10),
        message="Could not confirm this claim from retrieved sources.",
    )
    registry = _registry(
        FakeLLM(default={}),
        FakeSearch(),
        [
            FakeAgent(AgentName.EXTRACTOR),
            FakeAgent(AgentName.FACTCHECK, findings=[unverified]),
        ],
    )
    report = await Orchestrator(registry).review(OPINION_DOC, Profile.OPINION, ProgressRecorder())
    assert [f.id for f in report.unverified] == ["f_unverified"]
    assert report.must_fix == []
    assert report.verdict_line.startswith("Ready")  # unverified claims do not block
    assert "1 claim could not be verified." in report.verdict_line
