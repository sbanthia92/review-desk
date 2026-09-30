"""Real-pipeline wiring (lazy registry builder), fake mode and env parsing."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from reviewdesk.contracts import (
    AddClaim,
    AgentName,
    AgentResult,
    AuthError,
    Document,
    Message,
    MissingSetup,
    ModelTier,
    Profile,
    ProgressCallback,
    ProgressEvent,
    Report,
    ReviewContext,
)
from reviewdesk.mcp_local import (
    REGISTRY_BUILDER,
    REGISTRY_MODULE,
    EnvPipeline,
    RegistryPipeline,
    adapt_pipeline,
    create_server,
    default_runner,
)
from reviewdesk.mcp_local.demo import demo_pipeline, demo_registry
from reviewdesk.mcp_local.wiring import fake_mode, load_registry_builder, max_words_from_env
from reviewdesk.orchestrator import AgentRegistry
from reviewdesk.testing.fakes import (
    DESIGN_DOC,
    OPINION_DOC,
    FakeAgent,
    FakeFetcher,
    FakeLLM,
    FakeSearch,
    empty_report,
    sample_ledger,
    sample_report,
)
from tests.mcp_local.helpers import ProgressLog, call, text_of


def fake_registry(llm: FakeLLM | None = None) -> AgentRegistry:
    extractor = FakeAgent(
        AgentName.EXTRACTOR,
        ledger_updates=[AddClaim(claim=e.claim) for e in sample_ledger().entries.values()],
    )
    return AgentRegistry.of(
        [extractor, FakeAgent(AgentName.COPYEDIT)],
        llm=llm or FakeLLM(),
        search=FakeSearch(),
        fetcher=FakeFetcher(),
    )


@pytest.fixture
def registry_module() -> Iterator[types.ModuleType]:
    """Install a stand-in ``reviewdesk.registry`` module for one test."""
    module = types.ModuleType(REGISTRY_MODULE)
    saved = sys.modules.get(REGISTRY_MODULE)
    sys.modules[REGISTRY_MODULE] = module
    try:
        yield module
    finally:
        if saved is None:
            sys.modules.pop(REGISTRY_MODULE, None)
        else:
            sys.modules[REGISTRY_MODULE] = saved


def test_builder_name_is_the_documented_one() -> None:
    assert (REGISTRY_MODULE, REGISTRY_BUILDER) == ("reviewdesk.registry", "build_registry_from_env")


def test_missing_registry_module_is_missing_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, REGISTRY_MODULE, None)  # import raises ModuleNotFoundError
    with pytest.raises(MissingSetup):
        load_registry_builder()


def test_module_without_builder_is_missing_setup(registry_module: types.ModuleType) -> None:
    with pytest.raises(MissingSetup, match=REGISTRY_BUILDER):
        load_registry_builder()


async def test_env_pipeline_without_wiring_gives_missing_setup_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, REGISTRY_MODULE, None)
    server = create_server(pipeline=EnvPipeline(), fetcher=FakeFetcher())
    result = await call(server, {"content": OPINION_DOC.text})
    assert result.is_error
    assert "Missing setup: set ANTHROPIC_API_KEY" in text_of(result)


async def test_builder_raising_missing_setup_is_reported(
    registry_module: types.ModuleType,
) -> None:
    def build() -> AgentRegistry:
        raise MissingSetup("ANTHROPIC_API_KEY is not set")

    setattr(registry_module, REGISTRY_BUILDER, build)
    server = create_server(pipeline=EnvPipeline(), fetcher=FakeFetcher())
    result = await call(server, {"content": OPINION_DOC.text})
    assert result.is_error
    assert "Missing setup" in text_of(result)
    assert "ANTHROPIC_API_KEY is not set" in text_of(result)


async def test_env_pipeline_runs_the_real_orchestrator_on_the_built_registry(
    registry_module: types.ModuleType,
) -> None:
    builds: list[int] = []

    def build() -> AgentRegistry:
        builds.append(1)
        return fake_registry()

    setattr(registry_module, REGISTRY_BUILDER, build)
    server = create_server(pipeline=EnvPipeline(), fetcher=FakeFetcher())
    progress = ProgressLog()

    first = await call(server, {"content": OPINION_DOC.text}, progress=progress)
    second = await call(server, {"content": OPINION_DOC.text})

    assert not first.is_error and not second.is_error
    assert text_of(first).startswith("# Review Desk report")
    assert builds == [1]  # built once, then cached
    assert progress.events[-1][0] == 100.0
    percents = [p for p, _, _ in progress.events]
    assert percents == sorted(percents)


async def test_async_builder_is_awaited() -> None:
    async def build() -> AgentRegistry:
        return fake_registry()

    report = await RegistryPipeline(build)(OPINION_DOC, Profile.OPINION, lambda e: None)
    assert isinstance(report, Report)


async def test_builder_returning_the_wrong_type_is_missing_setup() -> None:
    with pytest.raises(MissingSetup):
        await RegistryPipeline(lambda: object())(OPINION_DOC, Profile.OPINION, lambda e: None)


async def test_failed_build_is_retried_next_time() -> None:
    attempts: list[int] = []

    def build() -> AgentRegistry:
        attempts.append(1)
        if len(attempts) == 1:
            raise MissingSetup("not yet")
        return fake_registry()

    pipeline = RegistryPipeline(build)
    with pytest.raises(MissingSetup):
        await pipeline(OPINION_DOC, Profile.OPINION, lambda e: None)
    await pipeline(OPINION_DOC, Profile.OPINION, lambda e: None)
    assert len(attempts) == 2


async def test_focus_reaches_the_agents() -> None:
    registry = fake_registry()
    await RegistryPipeline(lambda: registry)(
        OPINION_DOC, Profile.OPINION, lambda e: None, focus="the goal figures"
    )
    copyedit = registry.get(AgentName.COPYEDIT)
    assert isinstance(copyedit, FakeAgent)
    assert copyedit.runs and copyedit.runs[0].focus == "the goal figures"


class ProviderProbe:
    """An extractor that calls the LLM and search once and swallows their errors."""

    name = AgentName.EXTRACTOR

    async def run(self, ctx: ReviewContext) -> AgentResult:
        errors: list[str] = []
        try:
            await ctx.llm.complete(
                [Message(role="user", content="x")], model_tier=ModelTier.CHEAP, tag="probe.llm"
            )
        except AuthError as exc:
            errors.append(str(exc))
        try:
            await ctx.search.search("q")
        except AuthError as exc:
            errors.append(str(exc))
        return AgentResult(agent=AgentName.EXTRACTOR, error="; ".join(errors) or None)


async def test_rejected_llm_key_fails_the_review_instead_of_degrading() -> None:
    registry = AgentRegistry.of(
        [ProviderProbe()],
        llm=FakeLLM(default=AuthError("anthropic: API key rejected (HTTP 401)")),
        search=FakeSearch(),
        fetcher=FakeFetcher(),
    )
    server = create_server(pipeline=RegistryPipeline(lambda: registry), fetcher=FakeFetcher())
    result = await call(server, {"content": OPINION_DOC.text})
    assert result.is_error
    assert "Invalid or out-of-credit provider key" in text_of(result)


async def test_rejected_search_key_fails_the_review() -> None:
    registry = AgentRegistry.of(
        [ProviderProbe()],
        llm=FakeLLM(default="ok"),
        search=FakeSearch(error=AuthError("brave: API key rejected (HTTP 401)")),
        fetcher=FakeFetcher(),
    )
    with pytest.raises(AuthError, match="brave"):
        await RegistryPipeline(lambda: registry)(OPINION_DOC, Profile.OPINION, lambda e: None)


async def test_adapt_pipeline_drops_focus() -> None:
    seen: list[tuple[Document, Profile]] = []

    async def plain(doc: Document, profile: Profile, on_progress: ProgressCallback) -> Report:
        seen.append((doc, profile))
        on_progress(ProgressEvent(step="done", message="ok", percent=100))
        return empty_report()

    server = create_server(pipeline=adapt_pipeline(plain), fetcher=FakeFetcher())
    result = await call(server, {"content": OPINION_DOC.text, "focus": "x"})
    assert not result.is_error
    assert seen[0][1] is Profile.AUTO


# ---------------------------------------------------------------------------
# Env parsing and fake mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), (" YES ", True), ("on", True), ("0", False), ("", False)],
)
def test_fake_mode_flag(value: str, expected: bool) -> None:
    assert fake_mode({"REVIEWDESK_FAKE": value}) is expected
    assert fake_mode({}) is False


@pytest.mark.parametrize(
    ("value", "expected"), [("", 20_000), ("500", 500), ("abc", 20_000), ("-3", 20_000)]
)
def test_max_words_from_env(value: str, expected: int) -> None:
    assert max_words_from_env(20_000, {"REVIEWDESK_MAX_WORDS": value}) == expected


def test_default_runner_picks_demo_or_env() -> None:
    assert isinstance(default_runner({"REVIEWDESK_FAKE": "1"}), RegistryPipeline)
    assert not isinstance(default_runner({"REVIEWDESK_FAKE": "1"}), EnvPipeline)
    assert isinstance(default_runner({}), EnvPipeline)


@pytest.mark.parametrize("doc", [OPINION_DOC, DESIGN_DOC])
async def test_demo_pipeline_reviews_any_text_offline(doc: Document) -> None:
    progress = ProgressLog()
    server = create_server(pipeline=demo_pipeline(delay=0), fetcher=FakeFetcher())
    result = await call(server, {"content": doc.text}, progress=progress)
    assert not result.is_error
    text = text_of(result)
    assert "Demo mode" in text
    assert "## Appendix: claim ledger" in text
    assert progress.events[-1][0] == 100.0


async def test_demo_claims_round_trip_to_the_text() -> None:
    registry = demo_registry(delay=0)
    report = await RegistryPipeline(lambda: registry)(DESIGN_DOC, Profile.AUTO, lambda e: None)
    claims = [e.claim for e in report.ledger.entries.values()]
    assert claims
    for claim in claims:
        assert claim.span.text_of(DESIGN_DOC.text) == claim.text


async def test_demo_handles_text_without_sentences() -> None:
    report = await demo_pipeline(delay=0)(Document.from_text("ok"), Profile.OPINION, lambda e: None)
    assert isinstance(report, Report)


def test_sample_report_renders_via_server_types() -> None:
    # Guard: the helper's default report is the populated sample.
    assert sample_report().must_fix
