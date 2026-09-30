"""Shared HTTP plumbing for search adapters.

Each adapter subclasses ``HttpSearchClient`` and implements ``_build_request``
(what to send) and ``_parse`` (JSON to ``SearchResult``s). The base class
handles retries, error mapping, result cleanup and key hygiene: the key is
held in a ``Secret`` whose ``repr`` is masked and it never appears in error
messages.
"""

from __future__ import annotations

import html
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlsplit

import httpx

from reviewdesk.contracts import AuthError, ProviderError, RateLimitError, SearchResult
from reviewdesk.providers.search.retry import RetryPolicy, parse_retry_after

MAX_SNIPPET_CHARS = 500
MAX_TITLE_CHARS = 300
DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
USER_AGENT = "ReviewDesk/0.1 (document reviewer)"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


class Secret:
    """Holds an API key. ``repr`` and ``str`` are masked."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """Return the raw key (only for building request headers)."""
        return self._value

    def __repr__(self) -> str:
        return "Secret('***')"

    __str__ = __repr__


@dataclass(frozen=True)
class SearchRequest:
    """One HTTP request an adapter wants sent."""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str | int] | None = None
    json: dict[str, Any] | None = None

    def __repr__(self) -> str:
        # Headers carry the API key; never show them.
        return f"SearchRequest(method={self.method!r}, url={self.url!r})"


def clean_text(value: object, limit: int) -> str:
    """Strip tags and entities, collapse whitespace and truncate."""
    if not isinstance(value, str):
        return ""
    text = _WS_RE.sub(" ", html.unescape(_TAG_RE.sub("", value))).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def is_http_url(value: object) -> bool:
    """True for absolute ``http``/``https`` URLs with a host."""
    if not isinstance(value, str):
        return False
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc)


class HttpSearchClient(ABC):
    """Base ``SearchClient`` over a JSON HTTP API.

    ``transport`` lets tests inject ``httpx.MockTransport``; ``retry`` controls
    backoff (inject a no-op sleep in tests). Use as an async context manager
    or call ``aclose`` to release connections.
    """

    #: Provider name used in error messages and the comparison script.
    name: str = "search"
    #: Largest ``k`` the API accepts in one call.
    max_k: int = 20

    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        retry: RetryPolicy | None = None,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        base_url: str | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise AuthError(f"{self.name}: missing API key")
        self._key = Secret(api_key.strip())
        self._retry = retry or RetryPolicy()
        self._base_url = base_url
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            trust_env=False,
            follow_redirects=False,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(api_key=***)"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    @abstractmethod
    def _build_request(self, query: str, k: int) -> SearchRequest:
        """Return the HTTP request for ``query`` asking for ``k`` hits."""

    @abstractmethod
    def _parse(self, data: Any) -> list[SearchResult]:
        """Turn the decoded JSON body into results (ranks are reassigned)."""

    async def search(self, query: str, *, k: int = 5) -> list[SearchResult]:
        """Return at most ``k`` results for ``query``, best first."""
        query = query.strip()
        if not query or k < 1:
            return []
        request = self._build_request(query, min(k, self.max_k))
        data = await self._send(request)
        try:
            raw = self._parse(data)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ProviderError(f"{self.name}: unexpected response shape") from exc
        return _finalize(raw, k)

    async def _send(self, request: SearchRequest) -> Any:
        last_error: ProviderError | None = None
        for attempt in range(self._retry.attempts):
            retry_after: float | None = None
            try:
                response = await self._client.request(
                    request.method,
                    request.url,
                    headers=request.headers,
                    params=request.params,
                    json=request.json,
                )
            except httpx.TimeoutException:
                last_error = ProviderError(f"{self.name}: request timed out", retryable=True)
            except httpx.TransportError:
                last_error = ProviderError(f"{self.name}: connection failed", retryable=True)
            else:
                status = response.status_code
                if status in (401, 403):
                    raise AuthError(f"{self.name}: API key rejected (HTTP {status})")
                if status == 402:
                    raise AuthError(f"{self.name}: out of credit or plan limit (HTTP 402)")
                if status == 429:
                    retry_after = parse_retry_after(response.headers.get("Retry-After"))
                    last_error = RateLimitError(f"{self.name}: rate limited (HTTP 429)")
                elif status >= 500:
                    last_error = ProviderError(
                        f"{self.name}: server error (HTTP {status})", retryable=True
                    )
                elif status >= 400:
                    raise ProviderError(f"{self.name}: request rejected (HTTP {status})")
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise ProviderError(f"{self.name}: response was not JSON") from exc
            if attempt + 1 < self._retry.attempts:
                await self._retry.wait(attempt, retry_after)
        assert last_error is not None
        raise last_error

    def _url(self, default: str) -> str:
        return self._base_url or default


def _finalize(raw: list[SearchResult], k: int) -> list[SearchResult]:
    """Drop non-HTTP and duplicate URLs, cap at ``k``, and renumber ranks."""
    seen: set[str] = set()
    out: list[SearchResult] = []
    for hit in raw:
        if not is_http_url(hit.url) or hit.url in seen:
            continue
        seen.add(hit.url)
        out.append(hit.model_copy(update={"rank": len(out) + 1}))
        if len(out) >= k:
            break
    return out
