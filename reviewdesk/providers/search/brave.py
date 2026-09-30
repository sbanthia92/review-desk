"""Brave Search API adapter (``GET /res/v1/web/search``).

Auth: ``X-Subscription-Token`` header. Snippets arrive with ``<strong>``
highlighting, which is stripped.
"""

from __future__ import annotations

from typing import Any

from reviewdesk.contracts import SearchResult
from reviewdesk.providers.search.base import (
    MAX_SNIPPET_CHARS,
    MAX_TITLE_CHARS,
    HttpSearchClient,
    SearchRequest,
    clean_text,
)

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearch(HttpSearchClient):
    """``SearchClient`` backed by the Brave Search web endpoint."""

    name = "brave"
    max_k = 20

    def _build_request(self, query: str, k: int) -> SearchRequest:
        return SearchRequest(
            method="GET",
            url=self._url(BRAVE_URL),
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self._key.reveal(),
            },
            params={"q": query, "count": k},
        )

    def _parse(self, data: Any) -> list[SearchResult]:
        web = data.get("web") or {}
        results: list[SearchResult] = []
        for item in web.get("results") or []:
            if not isinstance(item, dict):
                continue
            published = item.get("page_age") or item.get("age")
            results.append(
                SearchResult(
                    url=str(item.get("url", "")),
                    title=clean_text(item.get("title"), MAX_TITLE_CHARS),
                    snippet=clean_text(item.get("description"), MAX_SNIPPET_CHARS),
                    published=published if isinstance(published, str) else None,
                )
            )
        return results
