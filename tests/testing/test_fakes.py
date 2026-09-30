import asyncio

import pytest
from pydantic import BaseModel

from reviewdesk.contracts import (
    Agent,
    AgentResult,
    Budget,
    BudgetExceeded,
    Debater,
    Evidence,
    FetchError,
    Message,
    ModelTier,
    Rechecker,
    SchemaError,
    SearchResult,
    Usage,
)
from reviewdesk.testing.fakes import (
    FakeAgent,
    FakeDevilsAdvocate,
    FakeFactChecker,
    FakeFetcher,
    FakeLLM,
    FakeLLMUnscripted,
    FakeSearch,
    ProgressRecorder,
    make_context,
)


class Answer(BaseModel):
    value: int


MSG = [Message(role="user", content="hello there")]


async def test_fake_llm_text_and_schema():
    llm = FakeLLM({"a": "plain text", "b": {"value": 3}, "c": '{"value": 4}'})
    r = await llm.complete(MSG, model_tier=ModelTier.CHEAP, tag="a")
    assert r.text == "plain text" and r.parsed is None
    assert r.usage.input_tokens == 2 and r.usage.llm_calls == 1
    r = await llm.complete(MSG, schema=Answer, model_tier=ModelTier.MID, tag="b")
    assert r.parsed == Answer(value=3)
    r = await llm.complete(MSG, schema=Answer, model_tier=ModelTier.MID, tag="c")
    assert r.parsed == Answer(value=4)
    assert [c.tag for c in llm.calls] == ["a", "b", "c"]


async def test_fake_llm_prefix_queue_default_and_errors():
    llm = FakeLLM({"agent": ["first", "second"]}, default="fallback")
    tags = ["agent.x", "agent.y", "agent.z", "other"]
    texts = [(await llm.complete(MSG, model_tier=ModelTier.CHEAP, tag=t)).text for t in tags]
    assert texts == ["first", "second", "second", "fallback"]

    with pytest.raises(FakeLLMUnscripted):
        await FakeLLM().complete(MSG, model_tier=ModelTier.CHEAP, tag="missing")
    with pytest.raises(SchemaError):
        await FakeLLM({"t": {"wrong": 1}}).complete(
            MSG, schema=Answer, model_tier=ModelTier.CHEAP, tag="t"
        )
    with pytest.raises(RuntimeError):
        await FakeLLM({"t": RuntimeError("boom")}).complete(
            MSG, model_tier=ModelTier.CHEAP, tag="t"
        )


async def test_fake_llm_callable_reply():
    llm = FakeLLM({"echo": lambda messages, schema: messages[-1].content.upper()})
    r = await llm.complete(MSG, model_tier=ModelTier.STRONG, tag="echo")
    assert r.text == "HELLO THERE"


async def test_fake_search_matching():
    hit = SearchResult(url="https://a", title="A")
    search = FakeSearch({"liverpool": [hit]}, default=[])
    assert await search.search("Liverpool points 2019") == [hit]
    assert await search.search("unrelated") == []
    assert search.queries == ["Liverpool points 2019", "unrelated"]
    with pytest.raises(RuntimeError):
        await FakeSearch(error=RuntimeError("down")).search("q")


async def test_fake_fetcher():
    fetcher = FakeFetcher({"https://a": "page text", "https://b": FetchError("bad")})
    page = await fetcher.fetch("https://a")
    assert page.text == "page text" and page.status == 200
    with pytest.raises(FetchError):
        await fetcher.fetch("https://b")
    with pytest.raises(FetchError):
        await fetcher.fetch("https://missing")


async def test_fake_agents_satisfy_protocols():
    ctx = make_context()
    agent = FakeAgent("copyedit", progress=["working"], usage=Usage(llm_calls=1))
    assert isinstance(agent, Agent)
    assert not isinstance(agent, Rechecker)
    result = await agent.run(ctx)
    assert result.agent == "copyedit"
    assert isinstance(ctx.emit_progress, ProgressRecorder)
    assert ctx.emit_progress.events[0].message == "working"

    fc = FakeFactChecker(recheck_result=AgentResult(agent="factcheck", error="x"))
    assert isinstance(fc, Rechecker) and isinstance(fc, Debater)
    ev = [Evidence(url="https://a", title="A", excerpt="e")]
    assert (await fc.recheck(ctx, "c1", ev)).error == "x"
    assert (await fc.respond(ctx, "c1", ev)).claim_id == "c1"
    da = FakeDevilsAdvocate()
    assert isinstance(da, Debater) and not isinstance(da, Rechecker)
    assert (await da.respond(ctx, "c1", ev)).agent == "devils_advocate"

    with pytest.raises(ValueError):
        await FakeAgent("x", raises=ValueError("boom")).run(ctx)


async def test_fake_agent_delay_runs_in_parallel():
    ctx = make_context()
    agents = [FakeAgent(f"a{i}", delay=0.05) for i in range(4)]
    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.gather(*(a.run(ctx) for a in agents))
    assert loop.time() - start < 0.15


def test_budget_meter():
    ctx = make_context(budget=Budget(max_tokens=10, max_search_calls=1, max_seconds=60))
    ctx.meter.require_search()
    ctx.meter.charge(Usage(search_calls=1, input_tokens=4))
    assert ctx.meter.searches_left() == 0 and ctx.meter.tokens_left() == 6
    assert ctx.meter.exhausted()
    with pytest.raises(BudgetExceeded):
        ctx.meter.require_search()
    with pytest.raises(BudgetExceeded):
        ctx.meter.require_tokens(7)
