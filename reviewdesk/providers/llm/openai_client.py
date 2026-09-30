"""OpenAI adapter implementing ``LLMClient`` via Chat Completions.

Structured output uses ``response_format={"type": "json_schema", ...}`` (not
strict, because arbitrary Pydantic schemas rarely satisfy strict mode's
constraints); the JSON is then validated against the Pydantic model, with one
repair re-ask on failure before raising ``SchemaError``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import openai
from openai.types.chat import ChatCompletion
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
    rate_limited,
    schema_error,
    schema_name,
    transient,
    validate,
    with_retries,
)
from reviewdesk.providers.llm.config import (
    DEFAULT_OPENAI_MODELS,
    DEFAULT_TIMEOUT_SECONDS,
    RetryPolicy,
    resolve_models,
)

if TYPE_CHECKING:
    import httpx2

PROVIDER = "openai"

NO_TEMPERATURE_PREFIXES: tuple[str, ...] = ("gpt-5", "gpt-6", "o1", "o3", "o4")
"""Reasoning-model families that reject a non-default ``temperature``."""


class OpenAILLM:
    """``LLMClient`` backed by the OpenAI Chat Completions API.

    The API key is passed in at construction, handed to the SDK and never
    logged or shown in ``repr``. ``temperature`` is dropped for models whose
    name starts with one of ``no_temperature_prefixes``. ``max_tokens`` maps to
    ``max_completion_tokens`` and is only sent when the caller passes it (or
    ``default_max_tokens`` is set), since reasoning models spend completion
    tokens on hidden reasoning.
    """

    provider = PROVIDER

    def __init__(
        self,
        api_key: str,
        *,
        models: Mapping[ModelTier | str, str] | None = None,
        retry: RetryPolicy | None = None,
        default_max_tokens: int | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        schema_retries: int = 1,
        no_temperature_prefixes: Sequence[str] = NO_TEMPERATURE_PREFIXES,
        base_url: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise AuthError("openai API key is empty")
        self.models = resolve_models(DEFAULT_OPENAI_MODELS, models)
        self.retry = retry or RetryPolicy()
        self.default_max_tokens = default_max_tokens
        self.schema_retries = schema_retries
        self.no_temperature_prefixes = tuple(no_temperature_prefixes)
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )

    def __repr__(self) -> str:
        return f"OpenAILLM(models={ {t.value: m for t, m in self.models.items()}!r})"

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
        convo: list[dict[str, Any]] = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict[str, Any] = {"model": model}
        cap = max_tokens or self.default_max_tokens
        if cap:
            kwargs["max_completion_tokens"] = cap
        if temperature is not None and not model.startswith(self.no_temperature_prefixes):
            kwargs["temperature"] = temperature

        if schema is None:
            resp = await self._create(messages=convo, **kwargs)
            choice = resp.choices[0] if resp.choices else None
            text = (choice.message.content or "") if choice else ""
            if choice is not None and not text and choice.message.refusal:
                raise ProviderError("openai model refused the request", retryable=False)
            return LLMResponse(text=text, model=resp.model or model, usage=_usage(resp))

        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name(schema),
                "description": (schema.__doc__ or "").strip()[:1024],
                "schema": json_schema_for(schema),
                "strict": False,
            },
        }

        total = Usage()
        reason = "no output"
        for _ in range(self.schema_retries + 1):
            resp = await self._create(messages=convo, **kwargs)
            total = total + _usage(resp)
            choice = resp.choices[0] if resp.choices else None
            if choice is None:
                reason = "no choices returned"
                continue
            if choice.message.refusal:
                reason = "model refused"
                break
            text = choice.message.content or ""
            parsed, feedback = validate(schema, text) if text else (None, "- <root>: empty")
            if parsed is not None:
                return LLMResponse(text=text, parsed=parsed, model=resp.model or model, usage=total)
            reason = "validation failed"
            if choice.finish_reason == "length":
                reason = "output truncated at max_tokens"
                break
            convo = [
                *convo,
                {"role": "assistant", "content": text or "(empty)"},
                {"role": "user", "content": REPAIR_PROMPT.format(feedback=feedback)},
            ]
        raise schema_error(PROVIDER, schema, reason)

    async def _create(self, **kwargs: Any) -> ChatCompletion:
        async def call() -> ChatCompletion:
            resp = await self._client.chat.completions.create(**kwargs)
            assert isinstance(resp, ChatCompletion)
            return resp

        return await with_retries(call, _classify, self.retry, provider=PROVIDER)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()


def _usage(resp: ChatCompletion) -> Usage:
    u = resp.usage
    if u is None:
        return Usage(llm_calls=1)
    return Usage(
        input_tokens=u.prompt_tokens or 0,
        output_tokens=u.completion_tokens or 0,
        llm_calls=1,
    )


def _error_code(exc: openai.APIStatusError) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        return code
    body = exc.body
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            for key in ("code", "type"):
                if isinstance(err.get(key), str):
                    return str(err[key])
    return "error"


def _classify(exc: Exception) -> ProviderError | TransientFailure | None:
    """Map OpenAI SDK exceptions to Review Desk errors (status and code only)."""
    if isinstance(exc, openai.APITimeoutError):
        return transient(ProviderError("openai request timed out", retryable=True))
    if isinstance(exc, openai.APIConnectionError):
        return transient(ProviderError("openai connection failed", retryable=True))
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        code = _error_code(exc)
        headers = exc.response.headers if exc.response is not None else None
        if status in (401, 403):
            return AuthError(f"openai rejected the API key ({status} {code})")
        if status == 429 and code in ("insufficient_quota", "billing_hard_limit_reached"):
            return AuthError("openai account has insufficient quota")
        if status == 429:
            return rate_limited(PROVIDER, headers)
        if status == 408 or status >= 500:
            return transient(
                ProviderError(f"openai server error ({status} {code})", retryable=True),
                headers,
            )
        return ProviderError(f"openai request failed ({status} {code})", retryable=False)
    if isinstance(exc, openai.OpenAIError):
        return ProviderError(f"openai client error ({type(exc).__name__})", retryable=False)
    return None
