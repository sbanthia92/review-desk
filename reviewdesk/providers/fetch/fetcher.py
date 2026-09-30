"""``HttpFetcher``: the production ``Fetcher``.

Safety rules:

- SSRF: every URL, including every redirect hop, passes ``ssrf.check_url``
  (scheme, credentials, port, resolved IPs). The default transport repeats
  the IP check at connect time (``guarded.GuardedTransport``).
- Redirects are followed by hand, at most ``max_redirects`` hops.
- The body is streamed and reading stops at ``max_bytes`` (counted after
  decompression, so compression bombs are capped too); the page is then
  extracted from what was read.
- Per-phase timeouts plus an overall deadline per fetch, both capped.
- Only HTML and text content types are extracted; anything else is refused.

Fetched text is untrusted data; this module extracts it and never acts on it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import TracebackType
from typing import Self
from urllib.parse import urljoin

import httpx

from reviewdesk.contracts import BlockedURLError, FetchedPage, FetchError
from reviewdesk.providers.fetch.extract import extract_readable, plain_text
from reviewdesk.providers.fetch.guarded import GuardedTransport
from reviewdesk.providers.fetch.ssrf import (
    DEFAULT_ALLOWED_PORTS,
    Resolver,
    check_url,
    system_resolver,
)
from reviewdesk.providers.search.retry import RetryPolicy, parse_retry_after

DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_MAX_TEXT_CHARS = 200_000
DEFAULT_TOTAL_TIMEOUT = 20.0
MAX_TOTAL_TIMEOUT = 60.0
MAX_PHASE_TIMEOUT = 30.0
DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
USER_AGENT = "ReviewDesk/0.1 (document reviewer; fact-checking fetcher)"

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
TEXT_TYPES = frozenset({"text/plain", "text/markdown", "text/csv", "application/json"})


@dataclass(frozen=True)
class _Body:
    status: int
    content_type: str
    charset: str | None
    data: bytes


@dataclass(frozen=True)
class _Redirect:
    location: str


def _find_blocked(exc: BaseException) -> BlockedURLError | None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, BlockedURLError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _clamp_timeout(timeout: httpx.Timeout | float) -> httpx.Timeout:
    t = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)

    def cap(value: float | None) -> float:
        return MAX_PHASE_TIMEOUT if value is None else min(value, MAX_PHASE_TIMEOUT)

    return httpx.Timeout(
        connect=cap(t.connect), read=cap(t.read), write=cap(t.write), pool=cap(t.pool)
    )


class HttpFetcher:
    """``Fetcher`` over httpx with SSRF protection and size/redirect caps.

    ``resolver`` does DNS for the SSRF checks (inject a fake in tests).
    ``transport`` defaults to a ``GuardedTransport`` using the same resolver;
    tests pass ``httpx.MockTransport``. ``retry`` covers timeouts, connection
    errors, 429 and 5xx (inject a no-op sleep in tests).
    """

    def __init__(
        self,
        *,
        resolver: Resolver = system_resolver,
        transport: httpx.AsyncBaseTransport | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        total_timeout: float = DEFAULT_TOTAL_TIMEOUT,
        allowed_ports: frozenset[int] | None = DEFAULT_ALLOWED_PORTS,
        retry: RetryPolicy | None = None,
        user_agent: str = USER_AGENT,
    ) -> None:
        if max_bytes < 1 or max_redirects < 0 or max_text_chars < 1:
            raise ValueError("caps must be positive")
        self._resolver = resolver
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects
        self._max_text_chars = max_text_chars
        self._total_timeout = min(max(total_timeout, 0.1), MAX_TOTAL_TIMEOUT)
        self._allowed_ports = allowed_ports
        self._retry = retry or RetryPolicy(attempts=2, base_delay=0.5, max_delay=2.0)
        self._client = httpx.AsyncClient(
            transport=transport if transport is not None else GuardedTransport(resolver),
            timeout=_clamp_timeout(timeout),
            follow_redirects=False,
            trust_env=False,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            },
        )

    def __repr__(self) -> str:
        return f"HttpFetcher(max_bytes={self._max_bytes}, max_redirects={self._max_redirects})"

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

    async def fetch(self, url: str) -> FetchedPage:
        """Fetch ``url`` and return its readable text.

        Raises ``BlockedURLError`` for URLs refused by SSRF rules and
        ``FetchError`` for every other failure.
        """
        try:
            async with asyncio.timeout(self._total_timeout):
                return await self._fetch(url)
        except TimeoutError:
            raise FetchError("fetch timed out") from None

    async def _fetch(self, url: str) -> FetchedPage:
        current = url.strip()
        for _hop in range(self._max_redirects + 1):
            await check_url(current, self._resolver, allowed_ports=self._allowed_ports)
            outcome = await self._get(current)
            if isinstance(outcome, _Redirect):
                current = urljoin(current, outcome.location)
                continue
            return self._to_page(url, current, outcome)
        raise FetchError(f"too many redirects (limit {self._max_redirects})")

    async def _get(self, url: str) -> _Body | _Redirect:
        last_error = FetchError("fetch failed")
        for attempt in range(self._retry.attempts):
            retry_after: float | None = None
            try:
                async with self._client.stream("GET", url) as response:
                    status = response.status_code
                    if status in REDIRECT_STATUSES:
                        location = response.headers.get("Location")
                        if not location:
                            raise FetchError(f"redirect without location (HTTP {status})")
                        return _Redirect(location.strip())
                    if status in RETRYABLE_STATUSES:
                        retry_after = parse_retry_after(response.headers.get("Retry-After"))
                        last_error = FetchError(f"HTTP {status}")
                    elif status >= 400:
                        raise FetchError(f"HTTP {status}")
                    elif not 200 <= status < 300:
                        raise FetchError(f"unexpected HTTP {status}")
                    else:
                        return await self._read_body(response)
            except httpx.TimeoutException:
                last_error = FetchError("fetch timed out")
            except httpx.TransportError as exc:
                blocked = _find_blocked(exc)
                if blocked is not None:
                    raise BlockedURLError(str(blocked)) from exc
                last_error = FetchError("could not connect")
            if attempt + 1 < self._retry.attempts:
                await self._retry.wait(attempt, retry_after)
        raise last_error

    async def _read_body(self, response: httpx.Response) -> _Body:
        header = response.headers.get("Content-Type", "")
        mime = header.split(";", 1)[0].strip().lower() or "text/html"
        if mime not in HTML_TYPES and mime not in TEXT_TYPES:
            raise FetchError(f"unsupported content type: {mime[:80]}")
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            remaining = self._max_bytes - size
            chunks.append(chunk[:remaining])
            size += min(len(chunk), remaining)
            if size >= self._max_bytes:
                break
        return _Body(
            status=response.status_code,
            content_type=mime,
            charset=response.charset_encoding,
            data=b"".join(chunks),
        )

    def _to_page(self, url: str, final_url: str, body: _Body) -> FetchedPage:
        if body.content_type in HTML_TYPES:
            title, text = extract_readable(body.data, encoding=body.charset)
        else:
            try:
                decoded = body.data.decode(body.charset or "utf-8", errors="replace")
            except LookupError:
                decoded = body.data.decode("utf-8", errors="replace")
            title, text = "", plain_text(decoded)
        return FetchedPage(
            url=url,
            final_url=final_url,
            status=body.status,
            title=title[:300],
            text=text[: self._max_text_chars],
            content_type=body.content_type,
        )
