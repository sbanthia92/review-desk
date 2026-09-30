"""CLI error messages and exit codes.

Messages are safe to print: they never contain an API key or document text.
Messages from Review Desk errors are included because the contract promises
they are safe (``reviewdesk.contracts.errors``); anything unexpected is
reduced to its type name.
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

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_SETUP = 3
EXIT_AUTH = 4
EXIT_TOO_LONG = 5
EXIT_FETCH = 6
EXIT_RATE_LIMIT = 7
EXIT_INPUT = 8
EXIT_INTERRUPTED = 130

MAX_DETAIL_CHARS = 300

SETUP_HINT = (
    "Set ANTHROPIC_API_KEY (or OPENAI_API_KEY) and one search key (BRAVE_SEARCH_API_KEY, "
    "TAVILY_API_KEY or EXA_API_KEY) in the environment or in a .env file "
    "(see .env.example; pass another file with --env-file)."
)


class InputError(ReviewDeskError):
    """The input file or URL argument is unusable (missing, unreadable, empty)."""


def _detail(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if len(text) > MAX_DETAIL_CHARS:
        text = text[: MAX_DETAIL_CHARS - 1] + "…"
    return text


def _with_detail(message: str, exc: BaseException) -> str:
    detail = _detail(exc)
    return f"{message} ({detail})" if detail else message


def too_long_message(words: int, limit: int) -> str:
    """The document-too-long message."""
    return (
        f"Document too long: {words:,} words; the limit is {limit:,} words. "
        "Review a section at a time, or raise the cap with --max-words."
    )


def describe(exc: BaseException, *, url: str | None = None) -> tuple[str, int]:
    """Map ``exc`` to ``(message, exit code)``."""
    if isinstance(exc, InputError):
        return f"Input error: {_detail(exc)}", EXIT_INPUT
    if isinstance(exc, MissingSetup):
        detail = _detail(exc)
        message = f"Missing setup: {detail}" if detail else "Missing setup."
        return f"{message}\n{SETUP_HINT}", EXIT_SETUP
    if isinstance(exc, DocumentTooLong):
        detail = _detail(exc)
        if not detail.startswith("Document too long"):
            detail = f"Document too long: {detail}"
        return detail, EXIT_TOO_LONG
    where = f" {url}" if url else ""
    if isinstance(exc, BlockedURLError):
        return (
            f"URL unreachable: refused to fetch{where}. Only public http(s) pages can be "
            "fetched (private, local and non-HTTP addresses are blocked). "
            "Save the text to a file and review that instead.",
            EXIT_FETCH,
        )
    if isinstance(exc, FetchError):
        return (
            _with_detail(f"URL unreachable: could not fetch{where}", exc)
            + ". Check the address, or save the text to a file and review that.",
            EXIT_FETCH,
        )
    if isinstance(exc, AuthError):
        return (
            _with_detail(
                "Invalid or out-of-credit provider key: the LLM or search provider rejected "
                "the API key",
                exc,
            )
            + ". Check the key and your account's credit.",
            EXIT_AUTH,
        )
    if isinstance(exc, RateLimitError):
        return (
            _with_detail("Rate limit exceeded: a provider is rate-limiting requests", exc)
            + ". Try again in a few minutes.",
            EXIT_RATE_LIMIT,
        )
    if isinstance(exc, ProviderError):
        hint = " Try again shortly." if exc.retryable else ""
        return _with_detail("A provider call failed", exc) + "." + hint, EXIT_FAILED
    if isinstance(exc, ReviewDeskError):
        return _with_detail("The review failed", exc), EXIT_FAILED
    return f"The review failed unexpectedly ({type(exc).__name__}).", EXIT_FAILED
