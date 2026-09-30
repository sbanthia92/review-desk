"""Web search adapters implementing ``SearchClient``.

Three providers with simple JSON HTTP APIs and a free or cheap entry tier:

- ``BraveSearch``: Brave Search API, an independent web index; plain
  keyword-style web results.
- ``TavilySearch``: Tavily, a search API built for LLM agents; returns
  cleaned content snippets.
- ``ExaSearch``: Exa, embedding-based search; good at finding pages that
  discuss a claim rather than pages matching its keywords.

Keys come in at construction (see ``config.PROVIDERS`` for env var names:
``BRAVE_SEARCH_API_KEY``, ``TAVILY_API_KEY``, ``EXA_API_KEY``). They are held
in a masked ``Secret`` and never appear in ``repr`` or error messages.

**Free-tier limits must be re-confirmed before the Phase 0 decision.** The
providers were chosen without live pricing research. Monthly quotas,
per-second rate limits and whether a card is required change often. Check
each provider's pricing page and write the numbers next to the output of
``scripts/compare_search.py``.
"""

from reviewdesk.providers.search.base import HttpSearchClient, Secret
from reviewdesk.providers.search.brave import BraveSearch
from reviewdesk.providers.search.config import (
    PROVIDERS,
    make_search_client,
    search_clients_from_env,
)
from reviewdesk.providers.search.exa import ExaSearch
from reviewdesk.providers.search.retry import RetryPolicy, no_sleep
from reviewdesk.providers.search.tavily import TavilySearch

__all__ = [
    "PROVIDERS",
    "BraveSearch",
    "ExaSearch",
    "HttpSearchClient",
    "RetryPolicy",
    "Secret",
    "TavilySearch",
    "make_search_client",
    "no_sleep",
    "search_clients_from_env",
]
