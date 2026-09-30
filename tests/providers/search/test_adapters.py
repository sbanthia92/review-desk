import json
from collections.abc import Callable

import httpx
import pytest

from reviewdesk.contracts import AuthError, ProviderError, RateLimitError, SearchClient
from reviewdesk.providers.search import (
    BraveSearch,
    ExaSearch,
    HttpSearchClient,
    RetryPolicy,
    TavilySearch,
)

KEY = "sk-test-SECRET-123"

BRAVE_BODY = {
    "web": {
        "results": [
            {
                "title": "Arsenal <strong>Invincibles</strong>",
                "url": "https://example.com/invincibles",
                "description": "Arsenal went <strong>unbeaten</strong> in 2003&ndash;04.",
                "page_age": "2024-05-01T00:00:00",
            },
            {"title": "Dup", "url": "https://example.com/invincibles", "description": "x"},
            {"title": "Bad scheme", "url": "javascript:alert(1)", "description": "x"},
            {"title": "Second", "url": "https://example.org/b", "description": "b"},
        ]
    }
}
TAVILY_BODY = {
    "results": [
        {"title": "T1", "url": "https://a.example/1", "content": "first", "score": 0.9},
        {"title": "T2", "url": "https://a.example/2", "content": "second", "score": 0.5},
    ]
}
EXA_BODY = {
    "results": [
        {
            "title": "E1",
            "url": "https://e.example/1",
            "publishedDate": "2023-01-01",
            "highlights": ["one", "two"],
        },
        {"title": "E2", "url": "https://e.example/2", "text": "plain text body"},
    ]
}

Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


class SleepLog:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def make(
    cls: type[HttpSearchClient], rec: Recorder, sleep: SleepLog | None = None, attempts: int = 3
) -> HttpSearchClient:
    policy = RetryPolicy(
        attempts=attempts, base_delay=0.5, max_delay=4.0, sleep=sleep or SleepLog()
    )
    return cls(KEY, transport=httpx.MockTransport(rec), retry=policy)


def ok(body: object) -> httpx.Response:
    return httpx.Response(200, json=body)


async def test_brave_request_and_parsing():
    rec = Recorder([ok(BRAVE_BODY)])
    client = make(BraveSearch, rec)
    results = await client.search("arsenal unbeaten", k=5)
    req = rec.requests[0]
    assert req.method == "GET"
    assert req.url.host == "api.search.brave.com"
    assert req.url.params["q"] == "arsenal unbeaten"
    assert req.url.params["count"] == "5"
    assert req.headers["X-Subscription-Token"] == KEY
    assert KEY not in str(req.url)
    assert [r.url for r in results] == ["https://example.com/invincibles", "https://example.org/b"]
    assert [r.rank for r in results] == [1, 2]
    assert results[0].title == "Arsenal Invincibles"
    assert results[0].snippet == "Arsenal went unbeaten in 2003–04."
    assert results[0].published == "2024-05-01T00:00:00"
    await client.aclose()


async def test_tavily_request_and_parsing():
    rec = Recorder([ok(TAVILY_BODY)])
    client = make(TavilySearch, rec)
    results = await client.search("q", k=1)
    req = rec.requests[0]
    assert req.method == "POST"
    assert req.url.host == "api.tavily.com"
    assert req.headers["Authorization"] == f"Bearer {KEY}"
    body = json.loads(req.content)
    assert body["query"] == "q"
    assert body["max_results"] == 1
    assert "api_key" not in body
    assert len(results) == 1
    assert results[0].snippet == "first"


async def test_exa_request_and_parsing():
    rec = Recorder([ok(EXA_BODY)])
    client = make(ExaSearch, rec)
    results = await client.search("q", k=3)
    req = rec.requests[0]
    assert req.url.host == "api.exa.ai"
    assert req.headers["x-api-key"] == KEY
    assert json.loads(req.content)["numResults"] == 3
    assert results[0].snippet == "one … two"
    assert results[0].published == "2023-01-01"
    assert results[1].snippet == "plain text body"


@pytest.mark.parametrize("cls", [BraveSearch, TavilySearch, ExaSearch])
async def test_k_caps_results_and_request(cls):
    rec = Recorder([ok({"web": {"results": []}, "results": []})])
    client = make(cls, rec)
    assert await client.search("q", k=500) == []
    req = rec.requests[0]
    sent = req.url.params.get("count") or json.loads(req.content or b"{}").get(
        "max_results", json.loads(req.content or b"{}").get("numResults")
    )
    assert int(sent) == cls.max_k


async def test_empty_query_and_zero_k_make_no_request():
    rec = Recorder([ok(BRAVE_BODY)])
    client = make(BraveSearch, rec)
    assert await client.search("   ") == []
    assert await client.search("q", k=0) == []
    assert rec.requests == []


@pytest.mark.parametrize("status", [401, 403, 402])
async def test_auth_errors_not_retried(status):
    rec = Recorder([httpx.Response(status, json={"error": "nope"})])
    sleep = SleepLog()
    client = make(TavilySearch, rec, sleep)
    with pytest.raises(AuthError) as info:
        await client.search("q")
    assert len(rec.requests) == 1
    assert sleep.delays == []
    assert KEY not in str(info.value)


async def test_rate_limit_retries_then_raises():
    rec = Recorder([httpx.Response(429, headers={"Retry-After": "2"})])
    sleep = SleepLog()
    client = make(BraveSearch, rec, sleep)
    with pytest.raises(RateLimitError) as info:
        await client.search("q")
    assert info.value.retryable is True
    assert len(rec.requests) == 3
    assert sleep.delays == [2.0, 2.0]


async def test_retry_after_is_capped():
    rec = Recorder([httpx.Response(429, headers={"Retry-After": "3600"}), ok(BRAVE_BODY)])
    sleep = SleepLog()
    client = make(BraveSearch, rec, sleep)
    assert await client.search("q")
    assert sleep.delays == [4.0]


async def test_server_error_then_success_with_backoff():
    rec = Recorder([httpx.Response(503), httpx.Response(500), ok(TAVILY_BODY)])
    sleep = SleepLog()
    client = make(TavilySearch, rec, sleep)
    results = await client.search("q")
    assert len(results) == 2
    assert sleep.delays == [0.5, 1.0]


async def test_server_error_exhausted_is_retryable_provider_error():
    rec = Recorder([httpx.Response(502)])
    client = make(ExaSearch, rec)
    with pytest.raises(ProviderError) as info:
        await client.search("q")
    assert not isinstance(info.value, RateLimitError | AuthError)
    assert info.value.retryable is True


@pytest.mark.parametrize(
    "exc", [httpx.ReadTimeout("slow"), httpx.ConnectError("down")], ids=["timeout", "connect"]
)
async def test_transport_errors_retried(exc):
    rec = Recorder([exc, ok(EXA_BODY)])
    client = make(ExaSearch, rec)
    assert len(await client.search("q")) == 2
    assert len(rec.requests) == 2


async def test_transport_errors_exhausted():
    rec = Recorder([httpx.ReadTimeout("slow")])
    client = make(ExaSearch, rec, attempts=2)
    with pytest.raises(ProviderError, match="timed out") as info:
        await client.search("q")
    assert info.value.retryable


async def test_bad_request_not_retried():
    rec = Recorder([httpx.Response(400, json={"error": "bad"})])
    client = make(BraveSearch, rec)
    with pytest.raises(ProviderError) as info:
        await client.search("q")
    assert info.value.retryable is False
    assert len(rec.requests) == 1


async def test_non_json_and_bad_shape():
    client = make(BraveSearch, Recorder([httpx.Response(200, text="<html>")]))
    with pytest.raises(ProviderError, match="not JSON"):
        await client.search("q")
    client = make(BraveSearch, Recorder([ok(["not", "a", "dict"])]))
    with pytest.raises(ProviderError, match="unexpected response shape"):
        await client.search("q")


def test_key_is_masked_and_required():
    client = BraveSearch(KEY)
    assert KEY not in repr(client)
    assert KEY not in str(client)
    assert KEY not in repr(client._key)
    assert KEY not in repr(vars(client))
    with pytest.raises(AuthError):
        TavilySearch("  ")


async def test_satisfies_protocol():
    client: SearchClient = make(BraveSearch, Recorder([ok(BRAVE_BODY)]))
    assert await client.search("q", k=1)


async def test_context_manager_closes():
    async with make(ExaSearch, Recorder([ok(EXA_BODY)])) as client:
        await client.search("q")
    assert client._client.is_closed


def test_retry_policy_validation_and_delay():
    with pytest.raises(ValueError):
        RetryPolicy(attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay=-1)
    policy = RetryPolicy(base_delay=1, max_delay=3)
    assert [policy.delay(i) for i in range(4)] == [1, 2, 3, 3]
    assert policy.delay(0, retry_after=0.25) == 0.25
