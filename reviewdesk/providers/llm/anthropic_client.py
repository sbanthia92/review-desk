"""Anthropic (Claude) adapter implementing ``LLMClient``.

Structured output uses a forced tool call: the Pydantic schema becomes the
tool's ``input_schema`` and the tool input is validated against the model. If
validation fails the adapter re-asks once (``schema_retries``) with the
validation errors as a ``tool_result``, then raises ``SchemaError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import anthropic
from pydantic import BaseModel

from reviewdesk.contracts import (
    AuthError,
    LLMResponse,
    Message,
    ModelTier,
    ProviderError,
    Usage,
)
from reviewdesk.providers.llm._common import (
    REPAIR_PROMPT,
    TransientFailure,
    json_schema_for,
    merge_consecutive,
    rate_limited,
    schema_error,
    schema_name,
    split_system,
    transient,
    validate,
    with_retries,
)
from reviewdesk.providers.llm.config import (
    DEFAULT_ANTHROPIC_MODELS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_SECONDS,
    RetryPolicy,
    resolve_models,
)

if TYPE_CHECKING:
    import httpx2

PROVIDER = "anthropic"

_BILLING_MARKERS = ("credit balance", "billing", "insufficient")


class AnthropicLLM:
    """``LLMClient`` backed by the Anthropic Messages API.

    The API key is passed in at construction, handed to the SDK and never
    logged or shown in ``repr``. ``http_client`` / ``base_url`` exist for tests
    and proxies. SDK-level retries are disabled; this class retries itself per
    ``retry``.

    ``temperature`` is not a parameter of the current Messages API (SDK 1.x
    removed it), so it is ignored unless ``send_temperature=True``, which
    passes it through ``extra_body`` for older models that still accept it.
    """

    provider = PROVIDER

    def __init__(
        self,
        api_key: str,
        *,
        models: Mapping[ModelTier | str, str] | None = None,
        retry: RetryPolicy | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        schema_retries: int = 1,
        send_temperature: bool = False,
        base_url: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise AuthError("anthropic API key is empty")
        self.models = resolve_models(DEFAULT_ANTHROPIC_MODELS, models)
        self.retry = retry or RetryPolicy()
        self.default_max_tokens = default_max_tokens
        self.schema_retries = schema_retries
        self.send_temperature = send_temperature
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )

    def __repr__(self) -> str:
        return f"AnthropicLLM(models={ {t.value: m for t, m in self.models.items()}!r})"

    __str__ = __repr__

    def model_for(self, tier: ModelTier) -> str:
        """The concrete model used for ``tier``."""
        return self.models[ModelTier(tier)]

    async def complete(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        model_tier: ModelTier,
        tag: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        model = self.model_for(model_tier)
        system, rest = split_system(messages)
        convo: list[dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in merge_consecutive(rest)
        ]
        if not convo:
            convo = [{"role": "user", "content": "Begin."}]

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self.default_max_tokens,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None and self.send_temperature:
            kwargs["extra_body"] = {"temperature": temperature}

        if schema is None:
            msg = await self._create(messages=convo, **kwargs)
            usage = _usage(msg)
            return LLMResponse(text=_text(msg), model=msg.model or model, usage=usage)

        tool = schema_name(schema)
        kwargs["tools"] = [
            {
                "name": tool,
                "description": (schema.__doc__ or f"Return a {schema.__name__}.").strip()[:1024],
                "input_schema": json_schema_for(schema),
            }
        ]
        kwargs["tool_choice"] = {"type": "tool", "name": tool}

        total = Usage()
        reason = "no output"
        for _ in range(self.schema_retries + 1):
            msg = await self._create(messages=convo, **kwargs)
            total = total + _usage(msg)
            block = next(
                (b for b in msg.content if b.type == "tool_use" and b.name == tool),
                None,
            )
            if block is None:
                reason = f"no tool call (stop_reason={msg.stop_reason})"
                if msg.stop_reason == "max_tokens":
                    break
                convo = [
                    *convo,
                    {"role": "assistant", "content": _text(msg) or "(no output)"},
                    {
                        "role": "user",
                        "content": f"Call the `{tool}` tool with your answer.",
                    },
                ]
                continue
            parsed, feedback = validate(schema, block.input)
            if parsed is not None:
                return LLMResponse(
                    text=_text(msg),
                    parsed=parsed,
                    model=msg.model or model,
                    usage=total,
                )
            reason = "validation failed"
            if msg.stop_reason == "max_tokens":
                reason = "output truncated at max_tokens"
                break
            convo = [
                *convo,
                {
                    "role": "assistant",
                    "content": _echo_blocks(msg),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "is_error": True,
                            "content": REPAIR_PROMPT.format(feedback=feedback),
                        }
                    ],
                },
            ]
        raise schema_error(PROVIDER, schema, reason)

    async def _create(self, **kwargs: Any) -> anthropic.types.Message:
        async def call() -> anthropic.types.Message:
            msg = await self._client.messages.create(**kwargs)
            assert isinstance(msg, anthropic.types.Message)
            return msg

        return await with_retries(call, _classify, self.retry, provider=PROVIDER)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()


def _text(msg: anthropic.types.Message) -> str:
    return "".join(b.text for b in msg.content if b.type == "text")


def _echo_blocks(msg: anthropic.types.Message) -> list[dict[str, Any]]:
    """Re-send the assistant turn as request params (text and tool_use only)."""
    out: list[dict[str, Any]] = []
    for b in msg.content:
        if b.type == "text" and b.text:
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


def _usage(msg: anthropic.types.Message) -> Usage:
    u = msg.usage
    cached = (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
    return Usage(
        input_tokens=(u.input_tokens or 0) + cached,
        output_tokens=u.output_tokens or 0,
        llm_calls=1,
    )


def _error_type(exc: anthropic.APIStatusError) -> str:
    body = exc.body
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("type"), str):
            return str(err["type"])
    return "error"


def _error_message(exc: anthropic.APIStatusError) -> str:
    body = exc.body
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return str(err["message"]).lower()
    return ""


def _classify(exc: Exception) -> ProviderError | TransientFailure | None:
    """Map Anthropic SDK exceptions to Review Desk errors.

    Messages carry only the status code and error type, never the SDK message
    (which could echo request details).
    """
    if isinstance(exc, anthropic.APITimeoutError):
        return transient(ProviderError("anthropic request timed out", retryable=True))
    if isinstance(exc, anthropic.APIConnectionError):
        return transient(ProviderError("anthropic connection failed", retryable=True))
    if isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code
        kind = _error_type(exc)
        headers = exc.response.headers if exc.response is not None else None
        if status in (401, 403):
            return AuthError(f"anthropic rejected the API key ({status} {kind})")
        if status == 400 and any(m in _error_message(exc) for m in _BILLING_MARKERS):
            return AuthError("anthropic account has insufficient credit")
        if status == 429:
            return rate_limited(PROVIDER, headers)
        if status == 408 or status >= 500:
            return transient(
                ProviderError(f"anthropic server error ({status} {kind})", retryable=True),
                headers,
            )
        return ProviderError(f"anthropic request failed ({status} {kind})", retryable=False)
    if isinstance(exc, anthropic.AnthropicError):
        return ProviderError(f"anthropic client error ({type(exc).__name__})", retryable=False)
    return None
