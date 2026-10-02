"""Tests for shared adapter helpers, config and the factory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reviewdesk.contracts import ModelTier
from reviewdesk.providers.llm import (
    DEFAULT_ANTHROPIC_MODELS,
    DEFAULT_OPENAI_MODELS,
    AnthropicLLM,
    OpenAILLM,
    RetryPolicy,
    make_llm_client,
)
from reviewdesk.providers.llm._common import json_schema_for, validate
from reviewdesk.providers.llm.config import resolve_models
from tests.providers.llm.replay import FAKE_KEY, FIXTURES
from tests.providers.llm.schemas import Cited, Tree, Verdict


def test_default_models_cover_every_tier() -> None:
    for mapping in (DEFAULT_ANTHROPIC_MODELS, DEFAULT_OPENAI_MODELS):
        assert set(mapping) == set(ModelTier)
    assert DEFAULT_ANTHROPIC_MODELS[ModelTier.CHEAP] == "claude-haiku-4-5-20251001"
    assert DEFAULT_ANTHROPIC_MODELS[ModelTier.MID] == "claude-sonnet-5-5"
    assert DEFAULT_ANTHROPIC_MODELS[ModelTier.STRONG] == "claude-opus-5-5"


def test_resolve_models_overrides() -> None:
    merged = resolve_models(DEFAULT_ANTHROPIC_MODELS, {"cheap": "x", ModelTier.MID: "y"})
    assert merged == {ModelTier.CHEAP: "x", ModelTier.MID: "y", ModelTier.STRONG: "claude-opus-5-5"}
    with pytest.raises(ValueError):
        resolve_models(DEFAULT_ANTHROPIC_MODELS, {"huge": "z"})


def test_retry_delay() -> None:
    policy = RetryPolicy(base_delay=1.0, max_delay=5.0, rng=lambda: 0.0)
    assert policy.delay(0) == 0.5
    assert policy.delay(1) == 1.0
    assert policy.delay(10) == 2.5  # capped at max_delay, half-jittered
    assert policy.delay(0, retry_after=3.0) == 3.0
    assert policy.delay(0, retry_after=60.0) == 5.0


def test_factory() -> None:
    assert isinstance(make_llm_client("anthropic", FAKE_KEY), AnthropicLLM)
    llm = make_llm_client("OpenAI", FAKE_KEY, models={"cheap": "gpt-4.1-nano"})
    assert isinstance(llm, OpenAILLM)
    assert llm.model_for(ModelTier.CHEAP) == "gpt-4.1-nano"
    with pytest.raises(ValueError):
        make_llm_client("mystery", FAKE_KEY)


def test_json_schema_inlines_refs() -> None:
    schema = json_schema_for(Cited)
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)
    assert schema["properties"]["sources"]["items"]["properties"]["url"]["type"] == "string"


def test_json_schema_keeps_recursive_defs() -> None:
    schema = json_schema_for(Tree)
    assert schema == Tree.model_json_schema()


def test_json_schema_plain() -> None:
    assert json_schema_for(Verdict) == Verdict.model_json_schema()


def test_validate_feedback_has_no_input_values() -> None:
    parsed, feedback = validate(Verdict, {"label": "SECRET-DOC-TEXT", "confidence": 2})
    assert parsed is None
    assert "label" in feedback and "confidence" in feedback
    assert "SECRET-DOC-TEXT" not in feedback
    parsed, feedback = validate(Verdict, '{"label": "wrong", "confidence": 0.1}')
    assert isinstance(parsed, Verdict) and feedback == ""


def test_fixtures_contain_no_secrets() -> None:
    for path in Path(FIXTURES).glob("*.json"):
        text = path.read_text()
        assert "sk-ant-" not in text, path.name
        assert "sk-proj-" not in text, path.name
        assert "x-api-key" not in json.loads(text).get("headers", {}), path.name
        assert "authorization" not in json.loads(text).get("headers", {}), path.name


def test_validate_decodes_json_encoded_list_fields() -> None:
    from pydantic import BaseModel

    from reviewdesk.providers.llm._common import unstringify, validate

    class Item(BaseModel):
        name: str

    class Out(BaseModel):
        items: list[Item]
        note: str = ""

    parsed, feedback = validate(Out, {"items": '[{"name": "a"}, {"name": "b"}]', "note": "[x"})
    assert feedback == ""
    assert isinstance(parsed, Out)
    assert [i.name for i in parsed.items] == ["a", "b"]
    assert parsed.note == "[x"  # not valid JSON: left as text

    same = {"items": [{"name": "a"}]}
    assert unstringify(same) is same
    parsed, feedback = validate(Out, {"items": "not a list"})
    assert parsed is None and "items" in feedback
