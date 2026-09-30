"""Tavily Search API adapter (``POST /search``).

Auth: ``Authorization: Bearer <key>``. Uses ``search_depth=basic`` (the
cheaper tier); ``content`` becomes the snippet.
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

TAVILY_URL = "https://api.tavily.com/search"


class TavilySearch(HttpSearchClient):
    """``SearchClient`` backed by Tavily's search endpoint."""

    name = "tavily"
    max_k = 20

    def _build_request(self, query: str, k: int) -> SearchRequest:
        return SearchRequest(
            method="POST",
            url=self._url(TAVILY_URL),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._key.reveal()}",
            },
            json={
                "query": query,
                "max_results": k,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            },
        )

    def _parse(self, data: Any) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in data.get("results") or []:
            if not isinstance(item, dict):
                continue
            published = item.get("published_date")
            results.append(
                SearchResult(
                    url=str(item.get("url", "")),
                    title=clean_text(item.get("title"), MAX_TITLE_CHARS),
                    snippet=clean_text(item.get("content"), MAX_SNIPPET_CHARS),
                    published=published if isinstance(published, str) else None,
                )
            )
        return results
