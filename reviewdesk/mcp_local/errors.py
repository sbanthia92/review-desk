"""Human-readable tool errors for the local MCP server.

Every message here is safe to show the user and the model: it never contains
an API key or document text. Exception messages from Review Desk errors are
included where the contract promises they are safe (``reviewdesk.contracts
.errors``); anything else is reduced to its type name.
"""

from __future__ import annotations

from reviewdesk.contracts import (
    AuthError,
    BlockedURLError,
    DocumentTooLong,
    FetchError,
    MissingSetup,
    ProviderError,
    RateLimitError,
    ReviewDeskError,
)

MAX_DETAIL_CHARS = 300

SETUP_HINT = (
    "set ANTHROPIC_API_KEY (or OPENAI_API_KEY) and a search key (BRAVE_SEARCH_API_KEY, "
    "TAVILY_API_KEY or EXA_API_KEY) in the environment of the Review Desk MCP server "
    "(the `env` block of your MCP client config), then restart the server. "
    "To try the server without keys, set REVIEWDESK_FAKE=1."
)


class InvalidInput(ReviewDeskError):
    """The tool arguments are unusable (both/neither of content and url, empty text)."""


def _detail(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if len(text) > MAX_DETAIL_CHARS:
        text = text[: MAX_DETAIL_CHARS - 1] + "…"
    return text


def _with_detail(message: str, exc: BaseException) -> str:
    detail = _detail(exc)
    return f"{message} ({detail})" if detail else message


def missing_setup_message(exc: MissingSetup | None = None) -> str:
    """The missing-setup message, with the builder's (safe) detail if any."""
    base = f"Missing setup: {SETUP_HINT}"
    if exc is None:
        return base
    detail = _detail(exc)
    return f"{base} Details: {detail}" if detail else base


def too_long_message(words: int, limit: int) -> str:
    """The document-too-long message."""
    return (
        f"Document too long: {words:,} words; the limit is {limit:,} words. "
        "Review a section at a time, or raise REVIEWDESK_MAX_WORDS."
    )


def user_message(exc: BaseException, *, url: str | None = None) -> str:
    """Map an exception from fetching or reviewing to a user-facing message."""
    if isinstance(exc, InvalidInput):
        return f"Invalid input: {_detail(exc)}"
    if isinstance(exc, MissingSetup):
        return missing_setup_message(exc)
    if isinstance(exc, DocumentTooLong):
        detail = _detail(exc)
        return detail if detail.startswith("Document too long") else f"Document too long: {detail}"
    if isinstance(exc, BlockedURLError):
        where = f" {url}" if url else ""
        return (
            f"URL unreachable: refused to fetch{where}. Only public http(s) pages can be "
            "fetched (private, local and non-HTTP addresses are blocked). "
            "Paste the text as `content` instead."
        )
    if isinstance(exc, FetchError):
        where = f" {url}" if url else ""
        return _with_detail(
            f"URL unreachable: could not fetch{where}. Check the address or paste the text "
            "as `content` instead",
            exc,
        )
    if isinstance(exc, AuthError):
        return _with_detail(
            "Invalid or out-of-credit provider key: the LLM or search provider rejected the "
            "API key. Check the key and your account's credit, then restart the server",
            exc,
        )
    if isinstance(exc, RateLimitError):
        return _with_detail(
            "Rate limit exceeded: a provider is rate-limiting requests. Try again in a few minutes",
            exc,
        )
    if isinstance(exc, ProviderError):
        hint = " Try again shortly." if exc.retryable else ""
        return _with_detail("A provider call failed", exc) + "." + hint
    if isinstance(exc, ReviewDeskError):
        return _with_detail("The review failed", exc)
    return (
        f"The review failed unexpectedly ({type(exc).__name__}). "
        "See the server's stderr log for details."
    )
