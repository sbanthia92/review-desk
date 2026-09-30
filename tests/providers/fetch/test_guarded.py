"""Connect-time SSRF checks (DNS rebinding) without touching the network."""

import httpcore
import pytest

from reviewdesk.contracts import BlockedURLError, FetchError
from reviewdesk.providers.fetch import GuardedNetworkBackend, HttpFetcher
from reviewdesk.providers.fetch.guarded import GuardedTransport
from reviewdesk.providers.fetch.ssrf import Resolver
from reviewdesk.providers.search import RetryPolicy, no_sleep

PUBLIC = "93.184.216.34"


class RecordingBackend(httpcore.AsyncNetworkBackend):
    """Inner backend that records connect targets and never opens sockets."""

    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.connects: list[tuple[str, int]] = []
        self.fail_for = fail_for or set()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):  # noqa: ASYNC109
        self.connects.append((host, port))
        if host in self.fail_for:
            raise httpcore.ConnectError("refused")
        return httpcore.AsyncMockStream([])

    async def sleep(self, seconds: float) -> None:
        return None


def resolver(mapping: dict[str, list[str]]) -> Resolver:
    async def resolve(host: str, port: int) -> list[str]:
        if host not in mapping:
            raise FetchError("could not resolve host")
        return mapping[host]

    return resolve


async def test_connects_to_vetted_ip_not_hostname():
    inner = RecordingBackend()
    backend = GuardedNetworkBackend(resolver({"example.com": [PUBLIC]}), inner)
    await backend.connect_tcp("example.com", 443)
    assert inner.connects == [(PUBLIC, 443)]


async def test_falls_back_to_next_address():
    inner = RecordingBackend(fail_for={PUBLIC})
    backend = GuardedNetworkBackend(resolver({"example.com": [PUBLIC, "8.8.8.8"]}), inner)
    await backend.connect_tcp("example.com", 80)
    assert inner.connects == [(PUBLIC, 80), ("8.8.8.8", 80)]


async def test_all_addresses_fail():
    inner = RecordingBackend(fail_for={PUBLIC})
    backend = GuardedNetworkBackend(resolver({"example.com": [PUBLIC]}), inner)
    with pytest.raises(httpcore.ConnectError):
        await backend.connect_tcp("example.com", 80)


@pytest.mark.parametrize("target", ["rebind.example", "127.0.0.1", "::ffff:10.0.0.1", "localhost"])
async def test_private_targets_never_connected(target):
    inner = RecordingBackend()
    backend = GuardedNetworkBackend(resolver({"rebind.example": ["127.0.0.1"]}), inner)
    with pytest.raises(httpcore.ConnectError) as info:
        await backend.connect_tcp(target, 80)
    assert isinstance(info.value.__cause__, BlockedURLError)
    assert inner.connects == []


async def test_unix_sockets_refused():
    backend = GuardedNetworkBackend(resolver({}), RecordingBackend())
    with pytest.raises(httpcore.ConnectError):
        await backend.connect_unix_socket("/var/run/docker.sock")


async def test_dns_rebinding_caught_at_connect_time():
    """The URL check sees a public IP; the connect-time lookup sees a private one."""
    answers = iter([[PUBLIC], ["127.0.0.1"]])

    async def flip(host: str, port: int) -> list[str]:
        return next(answers)

    inner = RecordingBackend()
    transport = GuardedTransport(flip, inner_backend=inner)
    fetcher = HttpFetcher(
        resolver=flip, transport=transport, retry=RetryPolicy(attempts=1, sleep=no_sleep)
    )
    with pytest.raises(BlockedURLError):
        await fetcher.fetch("http://rebind.example/")
    assert inner.connects == []
    await fetcher.aclose()


async def test_default_fetcher_uses_guarded_transport():
    async def to_loopback(host: str, port: int) -> list[str]:
        return ["127.0.0.1"]

    fetcher = HttpFetcher(resolver=to_loopback)
    assert isinstance(fetcher._client._transport, GuardedTransport)
    with pytest.raises(BlockedURLError):
        await fetcher.fetch("http://example.com/")
    await fetcher.aclose()
