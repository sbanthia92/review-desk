"""Exa search API adapter (``POST /search``).

Auth: ``x-api-key`` header. Requests ``type=auto`` with highlights, which
become the snippet (falling back to the first part of ``text``).
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

EXA_URL = "https://api.exa.ai/search"


class ExaSearch(HttpSearchClient):
    """``SearchClient`` backed by Exa's search endpoint."""

    name = "exa"
    max_k = 25

    def _build_request(self, query: str, k: int) -> SearchRequest:
        return SearchRequest(
            method="POST",
            url=self._url(EXA_URL),
            headers={"Accept": "application/json", "x-api-key": self._key.reveal()},
            json={
                "query": query,
                "numResults": k,
                "type": "auto",
                "contents": {"highlights": {"numSentences": 2, "highlightsPerUrl": 2}},
            },
        )

    def _parse(self, data: Any) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in data.get("results") or []:
            if not isinstance(item, dict):
                continue
            highlights = item.get("highlights")
            snippet_src: object
            if isinstance(highlights, list) and highlights:
                snippet_src = " … ".join(h for h in highlights if isinstance(h, str))
            else:
                snippet_src = item.get("text")
            published = item.get("publishedDate")
            results.append(
                SearchResult(
                    url=str(item.get("url", "")),
                    title=clean_text(item.get("title"), MAX_TITLE_CHARS),
                    snippet=clean_text(snippet_src, MAX_SNIPPET_CHARS),
                    published=published if isinstance(published, str) else None,
                )
            )
        return results
