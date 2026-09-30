"""Near-match detection between a sentence and untrusted web text.

The score is the share of the sentence's words that fall inside runs of at
least ``min_run`` consecutive words also found, in order, in the other text
(``difflib.SequenceMatcher`` over word tokens, case- and punctuation-
insensitive). 1.0 means the sentence appears verbatim; light edits keep the
score high; a shared topic with different wording scores near 0.

For long texts (fetched pages) the comparison runs on the window with the most
shared word trigrams, which keeps it fast and locates the excerpt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from urllib.parse import urlsplit

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")

MAX_EXCERPT_CHARS = 300
"""Excerpts longer than this are truncated with an ellipsis."""


@dataclass(frozen=True)
class _Token:
    word: str
    start: int
    end: int


@dataclass(frozen=True)
class Match:
    """Similarity of a sentence to some text, with the matching excerpt."""

    score: float
    excerpt: str


def _tokens(text: str) -> list[_Token]:
    return [
        _Token(m.group(0).lower().replace("’", "'"), m.start(), m.end())
        for m in _WORD.finditer(text)
    ]


def _best_window(sentence: list[str], other: list[str], size: int, n: int) -> tuple[int, int]:
    """``(start, end)`` of the ``size``-token window of ``other`` sharing most n-grams."""
    grams = {tuple(sentence[i : i + n]) for i in range(len(sentence) - n + 1)}
    hits = [i for i in range(len(other) - n + 1) if tuple(other[i : i + n]) in grams]
    if not hits:
        return 0, 0
    best_start, best_count, lo = hits[0], 0, 0
    for hi, pos in enumerate(hits):
        while pos - hits[lo] >= size:
            lo += 1
        if hi - lo + 1 > best_count:
            best_count, best_start = hi - lo + 1, hits[lo]
    start = max(0, best_start - len(sentence) // 2)
    return start, min(len(other), start + size)


def similarity(
    sentence: str, other: str, *, min_run: int = 3, max_direct_tokens: int = 400
) -> Match:
    """Score how closely ``other`` reproduces ``sentence`` (0.0 to 1.0)."""
    s_tokens = _tokens(sentence)
    o_tokens = _tokens(other)
    if not s_tokens or not o_tokens:
        return Match(0.0, "")
    s_words = [t.word for t in s_tokens]
    run = max(1, min(min_run, len(s_words)))
    if len(o_tokens) > max_direct_tokens:
        size = 2 * len(s_words) + 10
        lo, hi = _best_window(s_words, [t.word for t in o_tokens], size, run)
        o_tokens = o_tokens[lo:hi]
        if not o_tokens:
            return Match(0.0, "")
    o_words = [t.word for t in o_tokens]
    matcher = SequenceMatcher(None, s_words, o_words, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= run]
    if not blocks:
        return Match(0.0, "")
    matched = sum(b.size for b in blocks)
    first, last = blocks[0], blocks[-1]
    start = o_tokens[first.b].start
    end = o_tokens[last.b + last.size - 1].end
    return Match(round(matched / len(s_words), 4), _truncate(other[start:end]))


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_EXCERPT_CHARS:
        return text
    return text[: MAX_EXCERPT_CHARS - 1].rstrip() + "…"


def normalize_url(url: str) -> str:
    """Comparable form of a URL: no scheme, ``www.``, fragment or trailing slash."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower().removeprefix("www.")
    if not host:
        return url.strip().lower().rstrip("/")
    path = parts.path.rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{path}{query}"


def same_url(a: str | None, b: str | None) -> bool:
    """True if two URLs point at the same page (by ``normalize_url``)."""
    if not a or not b:
        return False
    return normalize_url(a) == normalize_url(b)
