"""Live tests against real providers. Deselected by default; run with
``uv run pytest -m live tests/providers/llm``. Each skips cleanly when its key
env var is unset. Keys are read from the environment and never printed.
"""

from __future__ import annotations

import os

import pytest

from reviewdesk.contracts import AuthError, Message, ModelTier
from reviewdesk.providers.llm import AnthropicLLM, OpenAILLM
from tests.providers.llm.schemas import Verdict

pytestmark = pytest.mark.live

MSGS = [
    Message(role="system", content="Answer briefly."),
    Message(role="user", content="Say the word 'ready' and nothing else."),
]

VERDICT_MSGS = [
    Message(role="system", content="You fact-check claims."),
    Message(
        role="user",
        content=(
            "Claim: 'Water boils at 100 degrees Celsius at sea level.' "
            "Return a verdict with a confidence and one short reason."
        ),
    ),
]


def _key(var: str) -> str:
    key = os.environ.get(var, "")
    if not key:
        pytest.skip(f"{var} not set")
    return key


@pytest.mark.parametrize(
    "cls,var", [(AnthropicLLM, "ANTHROPIC_API_KEY"), (OpenAILLM, "OPENAI_API_KEY")]
)
async def test_live_text(cls: type[AnthropicLLM] | type[OpenAILLM], var: str) -> None:
    llm = cls(_key(var))
    resp = await llm.complete(MSGS, model_tier=ModelTier.CHEAP, max_tokens=512)
    assert "ready" in resp.text.lower()
    assert resp.usage.input_tokens > 0
    assert resp.usage.output_tokens > 0
    assert resp.usage.llm_calls == 1


@pytest.mark.parametrize(
    "cls,var", [(AnthropicLLM, "ANTHROPIC_API_KEY"), (OpenAILLM, "OPENAI_API_KEY")]
)
async def test_live_structured(cls: type[AnthropicLLM] | type[OpenAILLM], var: str) -> None:
    llm = cls(_key(var))
    resp = await llm.complete(VERDICT_MSGS, schema=Verdict, model_tier=ModelTier.CHEAP)
    assert isinstance(resp.parsed, Verdict)
    assert resp.parsed.label == "verified"


@pytest.mark.parametrize("tier", [ModelTier.CHEAP, ModelTier.MID, ModelTier.STRONG])
async def test_live_anthropic_structured_every_tier(tier: ModelTier) -> None:
    """Every tier must accept the structured-output request, even with a small cap."""
    llm = AnthropicLLM(_key("ANTHROPIC_API_KEY"))
    resp = await llm.complete(VERDICT_MSGS, schema=Verdict, model_tier=tier, max_tokens=300)
    assert isinstance(resp.parsed, Verdict)
    assert resp.parsed.label == "verified"


@pytest.mark.parametrize(
    "cls,var", [(AnthropicLLM, "ANTHROPIC_API_KEY"), (OpenAILLM, "OPENAI_API_KEY")]
)
async def test_live_bad_key(cls: type[AnthropicLLM] | type[OpenAILLM], var: str) -> None:
    _key(var)  # only run when the provider is configured at all
    llm = cls("sk-invalid-key-for-testing")
    with pytest.raises(AuthError):
        await llm.complete(MSGS, model_tier=ModelTier.CHEAP, max_tokens=16)
