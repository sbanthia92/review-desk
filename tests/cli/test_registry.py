"""Real wiring from the environment: provider choice, errors, no key leaks."""

from __future__ import annotations

import logging

import pytest

from reviewdesk.agents.copyedit import CopyEditAgent
from reviewdesk.agents.devils_advocate import DevilsAdvocateAgent
from reviewdesk.agents.extractor import ExtractorAgent
from reviewdesk.agents.factcheck import FactCheckAgent
from reviewdesk.agents.originality import OriginalityAgent
from reviewdesk.agents.structure import StructureAgent
from reviewdesk.contracts import AgentName, MissingSetup, ModelTier
from reviewdesk.mcp_local.wiring import load_registry_builder
from reviewdesk.orchestrator import AgentRegistry
from reviewdesk.providers.fetch import HttpFetcher
from reviewdesk.providers.llm import AnthropicLLM, OpenAILLM
from reviewdesk.providers.search import PROVIDERS, BraveSearch, ExaSearch, TavilySearch
from reviewdesk.registry import (
    SEARCH_KEY_VARS,
    aclose_registry,
    build_registry,
    build_registry_from_env,
    default_agents,
    resolve_env_config,
)
from reviewdesk.testing.fakes import FakeFetcher, FakeLLM, FakeSearch

ANTHROPIC_KEY = "sk-ant-SECRET-anthropic-1234"
OPENAI_KEY = "sk-SECRET-openai-5678"
BRAVE_KEY = "brave-SECRET-9999"
TAVILY_KEY = "tvly-SECRET-0000"
EXA_KEY = "exa-SECRET-4242"
ALL_KEYS = [ANTHROPIC_KEY, OPENAI_KEY, BRAVE_KEY, TAVILY_KEY, EXA_KEY]

SIX = {
    AgentName.EXTRACTOR,
    AgentName.FACTCHECK,
    AgentName.DEVILS_ADVOCATE,
    AgentName.COPYEDIT,
    AgentName.STRUCTURE,
    AgentName.ORIGINALITY,
}


def _assert_no_keys(text: str) -> None:
    for key in ALL_KEYS:
        assert key not in text


async def _build(env: dict[str, str]) -> AgentRegistry:
    return build_registry_from_env(env)


async def test_anthropic_and_brave_by_default(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    env = {
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "OPENAI_API_KEY": OPENAI_KEY,
        "BRAVE_SEARCH_API_KEY": BRAVE_KEY,
        "EXA_API_KEY": EXA_KEY,
    }
    registry = await _build(env)
    try:
        assert isinstance(registry.llm, AnthropicLLM)
        assert isinstance(registry.search, BraveSearch)
        assert isinstance(registry.fetcher, HttpFetcher)
        assert set(registry.agents) == SIX
        assert isinstance(registry.get(AgentName.EXTRACTOR), ExtractorAgent)
        assert isinstance(registry.get(AgentName.FACTCHECK), FactCheckAgent)
        assert isinstance(registry.get(AgentName.DEVILS_ADVOCATE), DevilsAdvocateAgent)
        assert isinstance(registry.get(AgentName.COPYEDIT), CopyEditAgent)
        assert isinstance(registry.get(AgentName.STRUCTURE), StructureAgent)
        assert isinstance(registry.get(AgentName.ORIGINALITY), OriginalityAgent)
        _assert_no_keys(repr(registry))
        _assert_no_keys(str(registry))
        _assert_no_keys(caplog.text)
    finally:
        await aclose_registry(registry)


async def test_openai_when_only_openai_key() -> None:
    registry = await _build({"OPENAI_API_KEY": OPENAI_KEY, "TAVILY_API_KEY": TAVILY_KEY})
    try:
        assert isinstance(registry.llm, OpenAILLM)
        assert isinstance(registry.search, TavilySearch)
        _assert_no_keys(repr(registry))
    finally:
        await aclose_registry(registry)


async def test_explicit_provider_overrides() -> None:
    env = {
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "OPENAI_API_KEY": OPENAI_KEY,
        "BRAVE_SEARCH_API_KEY": BRAVE_KEY,
        "EXA_API_KEY": EXA_KEY,
        "REVIEWDESK_LLM_PROVIDER": " OpenAI ",
        "REVIEWDESK_SEARCH_PROVIDER": "exa",
    }
    registry = await _build(env)
    try:
        assert isinstance(registry.llm, OpenAILLM)
        assert isinstance(registry.search, ExaSearch)
    finally:
        await aclose_registry(registry)


async def test_model_overrides() -> None:
    env = {
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "BRAVE_SEARCH_API_KEY": BRAVE_KEY,
        "REVIEWDESK_MODEL_CHEAP": "claude-tiny",
        "REVIEWDESK_MODEL_STRONG": "claude-huge",
    }
    registry = await _build(env)
    try:
        assert isinstance(registry.llm, AnthropicLLM)
        assert registry.llm.model_for(ModelTier.CHEAP) == "claude-tiny"
        assert registry.llm.model_for(ModelTier.STRONG) == "claude-huge"
        assert registry.llm.model_for(ModelTier.MID)  # default kept
    finally:
        await aclose_registry(registry)


def test_config_repr_hides_keys() -> None:
    config = resolve_env_config({"ANTHROPIC_API_KEY": ANTHROPIC_KEY, "EXA_API_KEY": EXA_KEY})
    assert config.llm_provider == "anthropic"
    assert config.search_provider == "exa"
    _assert_no_keys(repr(config))


def _missing(env: dict[str, str]) -> str:
    with pytest.raises(MissingSetup) as info:
        build_registry_from_env(env)
    message = str(info.value)
    _assert_no_keys(message)
    return message


def test_missing_everything_names_all_variables() -> None:
    message = _missing({})
    for var in [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "TAVILY_API_KEY",
        "EXA_API_KEY",
    ]:
        assert var in message


def test_missing_llm_key_only() -> None:
    message = _missing({"BRAVE_SEARCH_API_KEY": BRAVE_KEY, "ANTHROPIC_API_KEY": "   "})
    assert "ANTHROPIC_API_KEY" in message
    assert "search" not in message


def test_missing_search_key_only() -> None:
    message = _missing({"ANTHROPIC_API_KEY": ANTHROPIC_KEY})
    assert "BRAVE_SEARCH_API_KEY" in message
    assert "LLM" not in message


def test_requested_provider_without_its_key() -> None:
    message = _missing(
        {
            "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
            "BRAVE_SEARCH_API_KEY": BRAVE_KEY,
            "REVIEWDESK_LLM_PROVIDER": "openai",
        }
    )
    assert "OPENAI_API_KEY" in message
    assert "REVIEWDESK_LLM_PROVIDER" in message


def test_unknown_provider_names() -> None:
    message = _missing(
        {
            "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
            "BRAVE_SEARCH_API_KEY": BRAVE_KEY,
            "REVIEWDESK_SEARCH_PROVIDER": "bing",
        }
    )
    assert "REVIEWDESK_SEARCH_PROVIDER" in message
    assert "brave" in message


def test_reads_os_environ_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", *SEARCH_KEY_VARS.values()]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("REVIEWDESK_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("REVIEWDESK_SEARCH_PROVIDER", raising=False)
    with pytest.raises(MissingSetup):
        build_registry_from_env()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("TAVILY_API_KEY", TAVILY_KEY)
    config = resolve_env_config()
    assert (config.llm_provider, config.search_provider) == ("openai", "tavily")


def test_search_key_vars_match_provider_table() -> None:
    assert {name: var for name, (var, _cls) in PROVIDERS.items()} == SEARCH_KEY_VARS


def test_default_agents_are_six_distinct() -> None:
    assert {agent.name for agent in default_agents()} == SIX


async def test_build_registry_with_explicit_clients() -> None:
    llm, search, fetcher = FakeLLM(), FakeSearch(), FakeFetcher()
    registry = build_registry(llm=llm, search=search, fetcher=fetcher)
    assert registry.llm is llm
    assert registry.search is search
    assert registry.fetcher is fetcher
    assert set(registry.agents) == SIX
    await aclose_registry(registry)  # fakes have no aclose: no error


async def test_build_registry_with_explicit_keys() -> None:
    registry = build_registry(
        llm_provider="openai",
        llm_api_key=OPENAI_KEY,
        search_provider="brave",
        search_api_key=BRAVE_KEY,
        fetcher=FakeFetcher(),
    )
    try:
        assert isinstance(registry.llm, OpenAILLM)
        assert isinstance(registry.search, BraveSearch)
        _assert_no_keys(repr(registry))
    finally:
        await aclose_registry(registry)


def test_build_registry_requires_clients_or_keys() -> None:
    with pytest.raises(MissingSetup):
        build_registry(search=FakeSearch(), fetcher=FakeFetcher())
    with pytest.raises(MissingSetup):
        build_registry(llm=FakeLLM(), fetcher=FakeFetcher())
    with pytest.raises(MissingSetup, match="unknown LLM provider"):
        build_registry(
            llm_provider="mistral", llm_api_key="x", search=FakeSearch(), fetcher=FakeFetcher()
        )


class _Closable:
    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


class _BrokenClose:
    async def aclose(self) -> None:
        raise RuntimeError("boom")


async def test_aclose_registry_closes_each_client_once() -> None:
    shared = _Closable()
    registry = AgentRegistry(llm=shared, search=shared, fetcher=_BrokenClose())  # type: ignore[arg-type]
    await aclose_registry(registry)
    assert shared.closed == 1


def test_mcp_wiring_finds_the_builder() -> None:
    builder = load_registry_builder()
    assert builder is build_registry_from_env
