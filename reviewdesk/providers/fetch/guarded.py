"""Connect-time SSRF enforcement for httpx.

``GuardedNetworkBackend`` wraps httpcore's network backend: every TCP
connect resolves the host through the injected ``Resolver``, requires every
address to be public, and connects to the vetted IP itself. TLS still uses
the original hostname (httpcore passes it as ``server_hostname``), so
certificate checks are unchanged. This closes the DNS-rebinding window
between the fetcher's URL check and the actual connection.

``guarded_transport`` builds an ``httpx.AsyncHTTPTransport`` that uses it.
"""

from __future__ import annotations

import ssl
from collections.abc import Iterable
from typing import Any

import httpcore
import httpx

from reviewdesk.contracts import BlockedURLError, FetchError
from reviewdesk.providers.fetch.ssrf import Resolver, resolve_public, system_resolver


class GuardedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Network backend that only connects to public IPs."""

    def __init__(
        self,
        resolver: Resolver = system_resolver,
        inner: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self._resolver = resolver
        self._inner = inner or httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 (httpcore signature)
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        try:
            addresses = await resolve_public(host, port, self._resolver)
        except BlockedURLError as exc:
            # httpcore only lets its own exceptions through cleanly; the
            # fetcher unwraps this back into BlockedURLError.
            raise httpcore.ConnectError(f"blocked: {exc}") from exc
        except FetchError as exc:
            raise httpcore.ConnectError(str(exc)) from exc
        last_error: Exception | None = None
        for address in addresses:
            try:
                return await self._inner.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,  # noqa: ASYNC109 (httpcore signature)
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise httpcore.ConnectError("blocked: unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class GuardedTransport(httpx.AsyncHTTPTransport):
    """``AsyncHTTPTransport`` whose connection pool uses ``GuardedNetworkBackend``.

    Proxies are never used (a proxy would bypass the IP checks).
    """

    def __init__(
        self,
        resolver: Resolver = system_resolver,
        *,
        inner_backend: httpcore.AsyncNetworkBackend | None = None,
        verify: ssl.SSLContext | bool = True,
        limits: httpx.Limits | None = None,
    ) -> None:
        limits = limits or httpx.Limits(max_connections=20, max_keepalive_connections=10)
        super().__init__(verify=verify, trust_env=False, limits=limits)
        ssl_context = verify if isinstance(verify, ssl.SSLContext) else None
        if ssl_context is None:
            ssl_context = httpx.create_ssl_context(verify=verify, trust_env=False)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_context,
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            network_backend=GuardedNetworkBackend(resolver, inner_backend),
        )


def guarded_transport(resolver: Resolver = system_resolver) -> GuardedTransport:
    """Default transport for ``HttpFetcher``."""
    return GuardedTransport(resolver)
