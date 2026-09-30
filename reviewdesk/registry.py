"""Real pipeline wiring: the six agents plus LLM, search and fetch providers.

``build_registry_from_env()`` is what the CLI (``reviewdesk review``) and the
local MCP server (``reviewdesk.mcp_local.wiring``) call. It reads:

| Variable | Meaning |
| --- | --- |
| ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` | LLM key (one is required) |
| ``BRAVE_SEARCH_API_KEY`` / ``TAVILY_API_KEY`` / ``EXA_API_KEY`` | search key (one is required) |
| ``REVIEWDESK_LLM_PROVIDER`` | ``anthropic`` or ``openai`` (default below) |
| ``REVIEWDESK_SEARCH_PROVIDER`` | ``brave``, ``tavily`` or ``exa`` (default below) |
| ``REVIEWDESK_MODEL_CHEAP`` / ``_MID`` / ``_STRONG`` | per-tier model overrides |

Default providers: whichever LLM key is set, Anthropic first; the first
search provider with a key, in the order brave, tavily, exa.

Missing keys raise ``MissingSetup`` naming the variables, never their values.
Nothing built here puts a key in its ``repr``; keys are never logged.

``build_registry(...)`` takes explicit clients (or provider names and keys)
for tests and embedding. ``aclose_registry`` closes the HTTP clients.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from reviewdesk.agents.copyedit import CopyEditAgent
from reviewdesk.agents.devils_advocate import DevilsAdvocateAgent
from reviewdesk.agents.extractor import ExtractorAgent
from reviewdesk.agents.factcheck import FactCheckAgent
from reviewdesk.agents.originality import OriginalityAgent
from reviewdesk.agents.structure import StructureAgent
from reviewdesk.contracts import (
    Agent,
    Fetcher,
    LLMClient,
    MissingSetup,
    ModelTier,
    SearchClient,
)
from reviewdesk.orchestrator import AgentRegistry

log = logging.getLogger(__name__)

__all__ = [
    "LLM_KEY_VARS",
    "LLM_PROVIDER_ENV",
    "MODEL_ENV",
    "SEARCH_KEY_VARS",
    "SEARCH_PROVIDER_ENV",
    "EnvConfig",
    "aclose_registry",
    "build_registry",
    "build_registry_from_env",
    "default_agents",
    "resolve_env_config",
]

LLM_PROVIDER_ENV = "REVIEWDESK_LLM_PROVIDER"
SEARCH_PROVIDER_ENV = "REVIEWDESK_SEARCH_PROVIDER"

LLM_KEY_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
"""LLM provider -> env var holding its key, in default preference order."""

SEARCH_KEY_VARS: dict[str, str] = {
    "brave": "BRAVE_SEARCH_API_KEY",
    "tavily": "TAVILY_API_KEY",
    "exa": "EXA_API_KEY",
}
"""Search provider -> env var holding its key, in default preference order.

Mirrors ``reviewdesk.providers.search.config.PROVIDERS`` (checked in tests).
"""

MODEL_ENV: dict[ModelTier, str] = {
    ModelTier.CHEAP: "REVIEWDESK_MODEL_CHEAP",
    ModelTier.MID: "REVIEWDESK_MODEL_MID",
    ModelTier.STRONG: "REVIEWDESK_MODEL_STRONG",
}
"""Per-tier model override variables."""


@dataclass(frozen=True)
class EnvConfig:
    """Provider choice resolved from the environment. Keys never show in ``repr``."""

    llm_provider: str
    search_provider: str
    llm_api_key: str = field(repr=False)
    search_api_key: str = field(repr=False)
    models: dict[ModelTier, str] = field(default_factory=dict)


def _get(env: Mapping[str, str], name: str) -> str:
    return (env.get(name) or "").strip()


def _choose(
    env: Mapping[str, str],
    *,
    kind: str,
    override_var: str,
    key_vars: Mapping[str, str],
    problems: list[str],
) -> tuple[str, str] | None:
    """Pick ``(provider, key)`` or append a safe problem description."""
    requested = _get(env, override_var).lower()
    if requested:
        if requested not in key_vars:
            known = ", ".join(key_vars)
            problems.append(f"{override_var} must be one of: {known}")
            return None
        key = _get(env, key_vars[requested])
        if not key:
            problems.append(
                f"{override_var}={requested} but {key_vars[requested]} is not set (no {kind} key)"
            )
            return None
        return requested, key
    for provider, var in key_vars.items():
        key = _get(env, var)
        if key:
            return provider, key
    problems.append(f"no {kind} key: set one of {', '.join(key_vars.values())}")
    return None


def resolve_env_config(env: Mapping[str, str] | None = None) -> EnvConfig:
    """Resolve providers, keys and model overrides from ``env`` (default ``os.environ``).

    Raises ``MissingSetup`` listing every missing variable (names only).
    """
    source: Mapping[str, str] = os.environ if env is None else env
    problems: list[str] = []
    llm = _choose(
        source,
        kind="LLM",
        override_var=LLM_PROVIDER_ENV,
        key_vars=LLM_KEY_VARS,
        problems=problems,
    )
    search = _choose(
        source,
        kind="search",
        override_var=SEARCH_PROVIDER_ENV,
        key_vars=SEARCH_KEY_VARS,
        problems=problems,
    )
    if llm is None or search is None:
        raise MissingSetup("Review Desk is not set up: " + "; ".join(problems) + ".")
    models = {tier: value for tier, var in MODEL_ENV.items() if (value := _get(source, var))}
    return EnvConfig(
        llm_provider=llm[0],
        llm_api_key=llm[1],
        search_provider=search[0],
        search_api_key=search[1],
        models=models,
    )


def default_agents() -> list[Agent]:
    """The six real agents with their default settings."""
    return [
        ExtractorAgent(),
        FactCheckAgent(),
        DevilsAdvocateAgent(),
        CopyEditAgent(),
        StructureAgent(),
        OriginalityAgent(),
    ]


def build_registry(
    *,
    llm: LLMClient | None = None,
    search: SearchClient | None = None,
    fetcher: Fetcher | None = None,
    agents: Iterable[Agent] | None = None,
    llm_provider: str | None = None,
    llm_api_key: str | None = None,
    search_provider: str | None = None,
    search_api_key: str | None = None,
    models: Mapping[ModelTier | str, str] | None = None,
) -> AgentRegistry:
    """Build a registry from explicit clients, or from provider names and keys.

    Pass ``llm`` or (``llm_provider`` and ``llm_api_key``); likewise ``search``
    or (``search_provider`` and ``search_api_key``). ``fetcher`` defaults to
    the SSRF-safe ``HttpFetcher`` and ``agents`` to ``default_agents()``.
    """
    if llm is None:
        if not llm_provider or not llm_api_key:
            raise MissingSetup("build_registry needs `llm` or `llm_provider` and `llm_api_key`")
        from reviewdesk.providers.llm import make_llm_client

        try:
            llm = make_llm_client(llm_provider, llm_api_key, models=models)
        except ValueError:
            known = ", ".join(LLM_KEY_VARS)
            raise MissingSetup(f"unknown LLM provider {llm_provider!r} (known: {known})") from None
    if search is None:
        if not search_provider or not search_api_key:
            raise MissingSetup(
                "build_registry needs `search` or `search_provider` and `search_api_key`"
            )
        from reviewdesk.providers.search import make_search_client

        search = make_search_client(search_provider, search_api_key)
    if fetcher is None:
        from reviewdesk.providers.fetch import HttpFetcher

        fetcher = HttpFetcher()
    chosen = list(agents) if agents is not None else default_agents()
    return AgentRegistry.of(chosen, llm=llm, search=search, fetcher=fetcher)


def build_registry_from_env(env: Mapping[str, str] | None = None) -> AgentRegistry:
    """The real registry wired from environment variables (default ``os.environ``).

    Raises ``MissingSetup`` with the names of missing variables when no LLM
    key or no search key is set.
    """
    config = resolve_env_config(env)
    overrides: dict[ModelTier | str, str] = {tier.value: m for tier, m in config.models.items()}
    registry = build_registry(
        llm_provider=config.llm_provider,
        llm_api_key=config.llm_api_key,
        search_provider=config.search_provider,
        search_api_key=config.search_api_key,
        models=overrides or None,
    )
    log.info(
        "registry built: llm=%s search=%s agents=%d",
        config.llm_provider,
        config.search_provider,
        len(registry.agents),
    )
    return registry


async def aclose_registry(registry: AgentRegistry) -> None:
    """Close the registry's LLM, search and fetch clients if they have ``aclose``.

    Errors while closing are logged (type only) and ignored.
    """
    seen: set[int] = set()
    for client in (registry.llm, registry.search, registry.fetcher):
        if id(client) in seen:
            continue
        seen.add(id(client))
        close = getattr(client, "aclose", None)
        if not callable(close):
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            log.debug("closing %s failed: %s", type(client).__name__, type(exc).__name__)
