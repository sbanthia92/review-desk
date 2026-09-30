import asyncio
import gzip
from collections.abc import AsyncIterator

import httpx
import pytest

from reviewdesk.contracts import BlockedURLError, Fetcher, FetchError
from reviewdesk.providers.fetch import HttpFetcher
from reviewdesk.providers.search import RetryPolicy

PUBLIC = "93.184.216.34"
HTML = "<html><head><title>T</title></head><body><p>Hello <b>world</b></p></body></html>"


async def no_sleep(_s: float) -> None:
    return None


def make_resolver(mapping: dict[str, list[str]] | None = None):
    mapping = mapping or {}
    calls: list[str] = []

    async def resolve(host: str, port: int) -> list[str]:
        calls.append(host)
        return mapping.get(host, [PUBLIC])

    resolve.calls = calls  # type: ignore[attr-defined]
    return resolve


Route = httpx.Response | Exception | list[httpx.Response | Exception]


class Server:
    """Routes by full URL; records requested URLs."""

    def __init__(self, routes: dict[str, Route]) -> None:
        self.routes = routes
        self.requested: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requested.append(url)
        route = self.routes.get(url)
        if route is None:
            return httpx.Response(404)
        if isinstance(route, list):
            route = route.pop(0) if len(route) > 1 else route[0]
        if isinstance(route, Exception):
            raise route
        return route


def html(body: str = HTML, status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(
        status,
        content=body.encode(),
        headers={"Content-Type": "text/html; charset=utf-8", **headers},
    )


def redirect(location: str, status: int = 302) -> httpx.Response:
    return httpx.Response(status, headers={"Location": location})


def make_fetcher(server: Server, resolver=None, **kwargs) -> HttpFetcher:
    kwargs.setdefault("retry", RetryPolicy(attempts=2, sleep=no_sleep))
    return HttpFetcher(
        resolver=resolver or make_resolver(),
        transport=httpx.MockTransport(server),
        **kwargs,
    )


async def test_fetches_and_extracts():
    server = Server({"https://example.com/a": html()})
    fetcher: Fetcher = make_fetcher(server)
    page = await fetcher.fetch("https://example.com/a")
    assert page.url == page.final_url == "https://example.com/a"
    assert page.status == 200
    assert page.title == "T"
    assert page.text == "Hello world"
    assert page.content_type == "text/html"


async def test_plain_text_content():
    server = Server(
        {
            "https://example.com/t": httpx.Response(
                200, text="line one\n\nline  two", headers={"Content-Type": "text/plain"}
            )
        }
    )
    page = await make_fetcher(server).fetch("https://example.com/t")
    assert page.text == "line one\nline two"
    assert page.content_type == "text/plain"


async def test_missing_content_type_treated_as_html():
    server = Server({"https://example.com/": httpx.Response(200, content=HTML.encode())})
    page = await make_fetcher(server).fetch("https://example.com/")
    assert page.text == "Hello world"


async def test_unsupported_content_type_refused():
    server = Server(
        {
            "https://example.com/x.pdf": httpx.Response(
                200, content=b"%PDF", headers={"Content-Type": "application/pdf"}
            )
        }
    )
    with pytest.raises(FetchError, match="unsupported content type"):
        await make_fetcher(server).fetch("https://example.com/x.pdf")


# --- SSRF --------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://localhost/",
    ],
)
async def test_blocked_urls_never_requested(url):
    server = Server({})
    with pytest.raises(BlockedURLError):
        await make_fetcher(server).fetch(url)
    assert server.requested == []


async def test_host_resolving_to_private_blocked_before_request():
    server = Server({"http://intranet.example/": html()})
    resolver = make_resolver({"intranet.example": ["192.168.0.10"]})
    with pytest.raises(BlockedURLError):
        await make_fetcher(server, resolver).fetch("http://intranet.example/")
    assert server.requested == []


async def test_redirect_to_private_ip_blocked():
    server = Server(
        {
            "https://example.com/start": redirect("http://169.254.169.254/latest/meta-data/"),
            "http://169.254.169.254/latest/meta-data/": html("secret"),
        }
    )
    with pytest.raises(BlockedURLError):
        await make_fetcher(server).fetch("https://example.com/start")
    assert server.requested == ["https://example.com/start"]


async def test_redirect_to_host_resolving_private_blocked():
    server = Server(
        {
            "https://example.com/start": redirect("https://rebind.example/x"),
            "https://rebind.example/x": html(),
        }
    )
    resolver = make_resolver({"rebind.example": ["10.0.0.5"]})
    with pytest.raises(BlockedURLError):
        await make_fetcher(server, resolver).fetch("https://example.com/start")
    assert server.requested == ["https://example.com/start"]
    assert resolver.calls == ["example.com", "rebind.example"]


async def test_redirect_to_non_http_scheme_blocked():
    server = Server({"https://example.com/start": redirect("file:///etc/passwd")})
    with pytest.raises(BlockedURLError):
        await make_fetcher(server).fetch("https://example.com/start")


async def test_blocked_port_on_redirect():
    server = Server({"https://example.com/start": redirect("http://example.com:6379/")})
    with pytest.raises(BlockedURLError, match="port"):
        await make_fetcher(server).fetch("https://example.com/start")


# --- redirects -----------------------------------------------------------------


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_follows_redirects_and_relative_locations(status):
    server = Server(
        {
            "https://example.com/a": redirect("/b", status),
            "https://example.com/b": redirect("https://other.example/c"),
            "https://other.example/c": html(),
        }
    )
    resolver = make_resolver()
    page = await make_fetcher(server, resolver).fetch("https://example.com/a")
    assert page.url == "https://example.com/a"
    assert page.final_url == "https://other.example/c"
    assert page.text == "Hello world"
    assert resolver.calls == ["example.com", "example.com", "other.example"]


async def test_redirect_cap():
    routes: dict[str, Route] = {
        f"https://example.com/{i}": redirect(f"/{i + 1}") for i in range(10)
    }
    server = Server(routes)
    with pytest.raises(FetchError, match="too many redirects"):
        await make_fetcher(server, max_redirects=3).fetch("https://example.com/0")
    assert len(server.requested) == 4


async def test_zero_redirects_allowed():
    server = Server({"https://example.com/0": redirect("/1")})
    with pytest.raises(FetchError, match="too many redirects"):
        await make_fetcher(server, max_redirects=0).fetch("https://example.com/0")


async def test_redirect_without_location():
    server = Server({"https://example.com/": httpx.Response(302)})
    with pytest.raises(FetchError, match="without location"):
        await make_fetcher(server).fetch("https://example.com/")


# --- size and time caps ------------------------------------------------------


async def test_body_truncated_at_byte_cap():
    chunks_sent: list[int] = []

    async def body() -> AsyncIterator[bytes]:
        for _ in range(1000):
            chunks_sent.append(1)
            yield b"<p>" + b"a" * 1000 + b"</p>"

    server = Server(
        {
            "https://example.com/big": httpx.Response(
                200, content=body(), headers={"Content-Type": "text/html"}
            )
        }
    )
    page = await make_fetcher(server, max_bytes=5000).fetch("https://example.com/big")
    assert 0 < len(page.text) <= 5000
    assert len(chunks_sent) < 20  # stopped reading early


async def test_byte_cap_counts_decompressed_bytes():
    payload = gzip.compress(b"<p>" + b"z" * 1_000_000 + b"</p>")
    server = Server(
        {
            "https://example.com/bomb": httpx.Response(
                200,
                content=payload,
                headers={"Content-Type": "text/html", "Content-Encoding": "gzip"},
            )
        }
    )
    page = await make_fetcher(server, max_bytes=10_000).fetch("https://example.com/bomb")
    assert len(page.text) <= 10_000


async def test_text_cap():
    server = Server({"https://example.com/": html("<p>" + "word " * 1000 + "</p>")})
    page = await make_fetcher(server, max_text_chars=100).fetch("https://example.com/")
    assert len(page.text) == 100


async def test_total_timeout():
    async def slow_resolver(host: str, port: int) -> list[str]:
        await asyncio.sleep(5)
        return [PUBLIC]

    server = Server({"https://example.com/": html()})
    fetcher = make_fetcher(server, slow_resolver, total_timeout=0.05)
    with pytest.raises(FetchError, match="timed out"):
        await fetcher.fetch("https://example.com/")


def test_timeouts_are_clamped():
    fetcher = HttpFetcher(timeout=httpx.Timeout(None), total_timeout=10_000)
    assert fetcher._total_timeout == 60.0
    t = fetcher._client.timeout
    assert (t.connect, t.read, t.write, t.pool) == (30.0, 30.0, 30.0, 30.0)


def test_invalid_caps():
    with pytest.raises(ValueError):
        HttpFetcher(max_bytes=0)


# --- status handling and retries --------------------------------------------


@pytest.mark.parametrize("status", [400, 403, 404, 410])
async def test_client_errors_not_retried(status):
    server = Server({"https://example.com/": [html(status=status)]})
    with pytest.raises(FetchError, match=f"HTTP {status}"):
        await make_fetcher(server).fetch("https://example.com/")
    assert len(server.requested) == 1


async def test_server_error_retried_then_succeeds():
    delays: list[float] = []

    async def sleep(s: float) -> None:
        delays.append(s)

    server = Server({"https://example.com/": [html(status=503, **{"Retry-After": "1"}), html()]})
    page = await make_fetcher(server, retry=RetryPolicy(attempts=2, sleep=sleep)).fetch(
        "https://example.com/"
    )
    assert page.text == "Hello world"
    assert delays == [1.0]


async def test_server_error_exhausted():
    server = Server({"https://example.com/": [html(status=500)]})
    with pytest.raises(FetchError, match="HTTP 500"):
        await make_fetcher(server).fetch("https://example.com/")
    assert len(server.requested) == 2


async def test_connect_errors_retried_then_fetch_error():
    server = Server({"https://example.com/": [httpx.ConnectError("refused")]})
    with pytest.raises(FetchError, match="could not connect") as info:
        await make_fetcher(server).fetch("https://example.com/")
    assert not isinstance(info.value, BlockedURLError)
    assert len(server.requested) == 2


async def test_read_timeout_retried():
    server = Server({"https://example.com/": [httpx.ReadTimeout("slow"), html()]})
    page = await make_fetcher(server).fetch("https://example.com/")
    assert page.text == "Hello world"


async def test_unexpected_status():
    server = Server({"https://example.com/": httpx.Response(304)})
    with pytest.raises(FetchError, match="unexpected HTTP 304"):
        await make_fetcher(server).fetch("https://example.com/")


async def test_context_manager_closes():
    server = Server({"https://example.com/": html()})
    async with make_fetcher(server) as fetcher:
        await fetcher.fetch("https://example.com/")
    assert fetcher._client.is_closed
