"""Build search clients from environment variables.

| Provider | Env var | Class |
| --- | --- | --- |
| ``brave`` | ``BRAVE_SEARCH_API_KEY`` | ``BraveSearch`` |
| ``tavily`` | ``TAVILY_API_KEY`` | ``TavilySearch`` |
| ``exa`` | ``EXA_API_KEY`` | ``ExaSearch`` |

Providers whose variable is unset or blank are skipped.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from reviewdesk.contracts import MissingSetup
from reviewdesk.providers.search.base import HttpSearchClient
from reviewdesk.providers.search.brave import BraveSearch
from reviewdesk.providers.search.exa import ExaSearch
from reviewdesk.providers.search.tavily import TavilySearch

PROVIDERS: dict[str, tuple[str, type[HttpSearchClient]]] = {
    "brave": ("BRAVE_SEARCH_API_KEY", BraveSearch),
    "tavily": ("TAVILY_API_KEY", TavilySearch),
    "exa": ("EXA_API_KEY", ExaSearch),
}
"""Provider name -> (env var holding its key, adapter class)."""


def make_search_client(name: str, api_key: str, **kwargs: Any) -> HttpSearchClient:
    """Build the adapter called ``name`` with ``api_key``."""
    if name not in PROVIDERS:
        known = ", ".join(PROVIDERS)
        raise MissingSetup(f"unknown search provider {name!r} (known: {known})")
    _env_var, cls = PROVIDERS[name]
    return cls(api_key, **kwargs)


def search_clients_from_env(
    env: Mapping[str, str] | None = None,
    *,
    only: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, HttpSearchClient]:
    """Build every provider whose key is set in ``env`` (default ``os.environ``).

    ``only`` restricts to the named providers; ``kwargs`` go to each adapter
    (e.g. ``transport`` or ``retry``).
    """
    source: Mapping[str, str] = os.environ if env is None else env
    names = only if only is not None else list(PROVIDERS)
    clients: dict[str, HttpSearchClient] = {}
    for name in names:
        if name not in PROVIDERS:
            known = ", ".join(PROVIDERS)
            raise MissingSetup(f"unknown search provider {name!r} (known: {known})")
        env_var, cls = PROVIDERS[name]
        key = source.get(env_var, "").strip()
        if key:
            clients[name] = cls(key, **kwargs)
    return clients
