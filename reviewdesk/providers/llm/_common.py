"""Helpers shared by the provider adapters (retry loop, schema handling)."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from reviewdesk.contracts import Message, ProviderError, RateLimitError, SchemaError
from reviewdesk.providers.llm.config import RetryPolicy

logger = logging.getLogger("reviewdesk.providers.llm")


class TransientFailure(Exception):
    """Internal: a retryable failure, carrying the error to raise if retries run out."""

    def __init__(self, error: ProviderError, retry_after: float | None = None) -> None:
        super().__init__(str(error))
        self.error = error
        self.retry_after = retry_after


async def with_retries[T](
    call: Callable[[], Awaitable[T]],
    classify: Callable[[Exception], ProviderError | TransientFailure | None],
    policy: RetryPolicy,
    *,
    provider: str,
) -> T:
    """Run ``call``, retrying transient failures per ``policy``.

    ``classify`` maps an SDK exception to a ``ProviderError`` (raised at once),
    a ``TransientFailure`` (retried) or None (re-raised unchanged).
    """
    attempt = 0
    while True:
        try:
            return await call()
        except Exception as exc:
            mapped = classify(exc)
            if mapped is None:
                raise
            if isinstance(mapped, ProviderError):
                raise mapped from None
            attempt += 1
            if attempt >= policy.max_attempts:
                raise mapped.error from None
            wait = policy.delay(attempt - 1, mapped.retry_after)
            logger.warning(
                "%s transient failure (%s); retry %d/%d in %.2fs",
                provider,
                type(mapped.error).__name__,
                attempt,
                policy.max_attempts - 1,
                wait,
            )
            await policy.sleep(wait)


def parse_retry_after(headers: Any) -> float | None:
    """Read a numeric ``retry-after`` header (seconds), if present."""
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def transient(error: ProviderError, headers: Any = None) -> TransientFailure:
    """Wrap ``error`` as retryable, reading any ``retry-after`` hint."""
    return TransientFailure(error, parse_retry_after(headers))


def rate_limited(provider: str, headers: Any = None) -> TransientFailure:
    """A retryable rate-limit failure."""
    return transient(RateLimitError(f"{provider} rate limit exceeded"), headers)


def split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Hoist system messages into one string; return it and the rest."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    rest = [m for m in messages if m.role != "system"]
    return system, rest


def merge_consecutive(messages: list[Message]) -> list[Message]:
    """Join consecutive messages with the same role (some APIs require alternation)."""
    merged: list[Message] = []
    for m in messages:
        if merged and merged[-1].role == m.role:
            merged[-1] = Message(role=m.role, content=f"{merged[-1].content}\n\n{m.content}")
        else:
            merged.append(m)
    return merged


def schema_name(schema: type[BaseModel]) -> str:
    """A provider-safe tool / schema name derived from the model class."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", schema.__name__)[:64]
    return name or "response"


def json_schema_for(schema: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for ``schema`` with non-recursive ``$ref``s inlined.

    Inlining makes the schema friendlier to providers with partial ``$ref``
    support. Recursive models keep their ``$defs``.
    """
    raw = schema.model_json_schema()
    defs: dict[str, Any] = raw.get("$defs", {})
    if not defs:
        return raw

    def inline(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref.removeprefix("#/$defs/")
                if key in stack or key not in defs:
                    raise _Recursive
                extra = {k: v for k, v in node.items() if k != "$ref"}
                target = inline(defs[key], (*stack, key))
                return {**target, **inline(extra, stack)} if extra else target
            return {k: inline(v, stack) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [inline(v, stack) for v in node]
        return node

    try:
        result: dict[str, Any] = inline(raw, ())
    except _Recursive:
        return raw
    return result


class _Recursive(Exception):
    pass


def validate(schema: type[BaseModel], data: Any) -> tuple[BaseModel | None, str]:
    """Validate ``data`` against ``schema``.

    Returns ``(instance, "")`` or ``(None, feedback)`` where ``feedback`` lists
    field locations and messages (no input values) for a repair prompt.
    """
    try:
        if isinstance(data, str | bytes):
            return schema.model_validate_json(data), ""
        return schema.model_validate(data), ""
    except ValidationError as exc:
        lines = [
            f"- {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors(include_input=False, include_url=False)[:20]
        ]
        return None, "\n".join(lines)


def schema_error(provider: str, schema: type[BaseModel], reason: str) -> SchemaError:
    """A ``SchemaError`` whose message is safe to show (no document text)."""
    return SchemaError(f"{provider} output did not match {schema.__name__}: {reason}")


REPAIR_PROMPT = (
    "Your previous response did not validate against the required schema:\n"
    "{feedback}\n"
    "Respond again with output that matches the schema exactly."
)
