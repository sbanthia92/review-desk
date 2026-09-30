"""Offline tests for ``OpenAILLM`` against recorded HTTP fixtures."""

from __future__ import annotations

import logging

import httpx2
import pytest

from reviewdesk.contracts import (
    AuthError,
    LLMClient,
    Message,
    ModelTier,
    ProviderError,
    RateLimitError,
    SchemaError,
)
from reviewdesk.providers.llm import OpenAILLM
from tests.providers.llm.replay import FAKE_KEY, Replay, Sleeps, fast_retry
from tests.providers.llm.schemas import Verdict

MSGS = [
    Message(role="system", content="You are a careful reviewer."),
    Message(role="user", content="Summarise the draft."),
]


def make(replay: Replay, sleeps: Sleeps | None = None, **kw: object) -> OpenAILLM:
    return OpenAILLM(
        FAKE_KEY,
        retry=fast_retry(sleeps or Sleeps()),
        http_client=replay.client(),
        **kw,  # type: ignore[arg-type]
    )


def test_satisfies_protocol() -> None:
    client: LLMClient = make(Replay())
    assert client is not None


async def test_text_completion_and_usage() -> None:
    replay = Replay("openai_text")
    resp = await make(replay).complete(MSGS, model_tier=ModelTier.CHEAP, tag="t.text")
    assert resp.text.startswith("The document argues")
    assert resp.parsed is None
    assert resp.model == "gpt-5.4-mini"
    assert resp.usage.input_tokens == 55
    assert resp.usage.output_tokens == 21
    assert resp.usage.llm_calls == 1

    body = replay.body()
    assert body["model"] == "gpt-5.4-mini"
    assert body["messages"] == [
        {"role": "system", "content": "You are a careful reviewer."},
        {"role": "user", "content": "Summarise the draft."},
    ]
    assert "max_completion_tokens" not in body
    assert "t.text" not in replay.requests[0].content.decode()


async def test_options_and_temperature_policy() -> None:
    replay = Replay("openai_text", "openai_text")
    llm = make(replay, models={ModelTier.MID: "gpt-4.1"})
    await llm.complete(MSGS, model_tier=ModelTier.MID, max_tokens=300, temperature=0.3)
    body = replay.body()
    assert body["model"] == "gpt-4.1"
    assert body["max_completion_tokens"] == 300
    assert body["temperature"] == 0.3
    # Reasoning-model families reject temperature, so it is dropped.
    await llm.complete(MSGS, model_tier=ModelTier.STRONG, temperature=0.3)
    body = replay.body()
    assert body["model"] == "gpt-5.5"
    assert "temperature" not in body


async def test_structured_output_via_json_schema() -> None:
    replay = Replay("openai_json_schema")
    resp = await make(replay).complete(MSGS, schema=Verdict, model_tier=ModelTier.MID)
    assert isinstance(resp.parsed, Verdict)
    assert resp.parsed.reasons == ["Primary source matches the figure."]
    rf = replay.body()["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Verdict"
    assert rf["json_schema"]["schema"]["properties"]["label"]["enum"] == [
        "verified",
        "wrong",
        "unsupported",
    ]


async def test_schema_repair_then_success() -> None:
    replay = Replay("openai_json_invalid", "openai_json_schema")
    resp = await make(replay).complete(MSGS, schema=Verdict, model_tier=ModelTier.MID)
    assert isinstance(resp.parsed, Verdict)
    assert resp.usage.llm_calls == 2
    assert resp.usage.input_tokens == 110
    assistant, feedback = replay.body()["messages"][-2:]
    assert assistant == {"role": "assistant", "content": '{"label": "maybe", "confidence": "high"}'}
    assert feedback["role"] == "user"
    assert "confidence" in feedback["content"]


async def test_schema_error_after_repair_fails() -> None:
    replay = Replay("openai_json_invalid", "openai_json_invalid")
    with pytest.raises(SchemaError):
        await make(replay).complete(MSGS, schema=Verdict, model_tier=ModelTier.MID)
    assert len(replay.requests) == 2


@pytest.mark.parametrize(
    ("fixture", "reason"),
    [("openai_truncated", "max_tokens"), ("openai_refusal", "refused")],
)
async def test_schema_error_without_repair(fixture: str, reason: str) -> None:
    replay = Replay(fixture)
    with pytest.raises(SchemaError, match=reason):
        await make(replay).complete(MSGS, schema=Verdict, model_tier=ModelTier.MID)
    assert len(replay.requests) == 1


async def test_refusal_without_schema() -> None:
    with pytest.raises(ProviderError, match="refused"):
        await make(Replay("openai_refusal")).complete(MSGS, model_tier=ModelTier.CHEAP)


async def test_retries_rate_limit_and_server_error() -> None:
    sleeps = Sleeps()
    replay = Replay("openai_rate_limit", "openai_server_error", "openai_text")
    resp = await make(replay, sleeps).complete(MSGS, model_tier=ModelTier.CHEAP)
    assert resp.usage.llm_calls == 1
    assert sleeps.waits == [1.0, 1.0]  # retry-after 1s, then 0.5 * 2**1


async def test_rate_limit_exhausted() -> None:
    replay = Replay(*["openai_rate_limit"] * 3)
    with pytest.raises(RateLimitError):
        await make(replay).complete(MSGS, model_tier=ModelTier.CHEAP)
    assert len(replay.requests) == 3


async def test_timeout_retried() -> None:
    sleeps = Sleeps()
    replay = Replay(httpx2.ReadTimeout("slow"), "openai_text")
    resp = await make(replay, sleeps).complete(MSGS, model_tier=ModelTier.CHEAP)
    assert resp.text
    assert len(sleeps.waits) == 1


@pytest.mark.parametrize("fixture", ["openai_auth_error", "openai_insufficient_quota"])
async def test_auth_errors_not_retried(fixture: str) -> None:
    sleeps = Sleeps()
    replay = Replay(fixture)
    with pytest.raises(AuthError) as info:
        await make(replay, sleeps).complete(MSGS, model_tier=ModelTier.CHEAP)
    assert sleeps.waits == []
    assert FAKE_KEY not in str(info.value)


async def test_key_sent_but_never_exposed(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    replay = Replay("openai_rate_limit", "openai_text")
    llm = make(replay)
    assert FAKE_KEY not in repr(llm)
    assert FAKE_KEY not in str(llm)
    await llm.complete(MSGS, model_tier=ModelTier.CHEAP)
    assert replay.requests[0].headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in caplog.text


def test_empty_key_rejected() -> None:
    with pytest.raises(AuthError):
        OpenAILLM("")
