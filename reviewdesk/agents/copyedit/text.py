"""Text helpers shared by the copy editor and structure reviewer (T6).

- ``TextLocator``: finds an LLM-quoted passage in the original text and returns
  spans into it (models are unreliable at offsets, so agents ask for verbatim
  quotes and locate them here). Adapted from the extractor's locator; this
  copy has no elision matching, because a copy-edit span must cover exactly
  the quoted passage.
- ``split_into_chunks``: contiguous, gap-free chunks for long documents.
- ``headings`` / ``title_span``: markdown heading spans, used to anchor
  structure findings about missing content.
- ``delimit``: wraps untrusted text in nonce-carrying markers.

Every span returned here covers a slice of the original text, so callers take
the passage from ``text[span.start:span.end]`` and it round-trips exactly.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from enum import StrEnum

from reviewdesk.contracts.models import Span

# ---------------------------------------------------------------------------
# Quote location
# ---------------------------------------------------------------------------

_CHAR_MAP: dict[str, str] = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "′": "'",
    "´": "'",
    "`": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "″": '"',
    "«": '"',
    "»": '"',
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    "…": "...",
}
"""Characters mapped to an ASCII equivalent before normalized matching."""

_DROP = frozenset("​‌‍⁠﻿­*_")
"""Characters ignored by normalized matching (zero-width, soft hyphen, emphasis)."""

_TRIM_CHARS = " .,;:!?'\"-"


class MatchMethod(StrEnum):
    """How a quote was located."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    TRIMMED = "trimmed"


@dataclass(frozen=True)
class Located:
    """Every occurrence of a quote, in document order, and how it was found."""

    spans: list[Span]
    method: MatchMethod


def normalize(text: str) -> tuple[str, list[int]]:
    """Normalize ``text`` for fuzzy matching.

    Returns the normalized string and, for each of its characters, the index
    of the original character it came from. Whitespace runs collapse to one
    space; leading whitespace is dropped; letters are lower-cased.
    """
    out: list[str] = []
    index: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        if ch in _DROP:
            continue
        rep = _CHAR_MAP.get(ch, ch)
        if rep.isspace():
            if not prev_space:
                out.append(" ")
                index.append(i)
                prev_space = True
            continue
        for r in rep:
            low = r.lower()
            out.append(low if len(low) == 1 else r)
            index.append(i)
        prev_space = False
    return "".join(out), index


def _find_all(haystack: str, needle: str) -> list[int]:
    """Start offsets of every (possibly overlapping) occurrence of ``needle``."""
    starts: list[int] = []
    if not needle:
        return starts
    pos = haystack.find(needle)
    while pos != -1:
        starts.append(pos)
        pos = haystack.find(needle, pos + 1)
    return starts


class TextLocator:
    """Finds quotes in one text. Build once per text; ``locate`` is reusable."""

    def __init__(self, text: str) -> None:
        self.text = text
        self._norm, self._index = normalize(text)

    def _normalized_spans(self, nquote: str) -> list[Span]:
        spans: list[Span] = []
        for s in _find_all(self._norm, nquote):
            start = self._index[s]
            end = self._index[s + len(nquote) - 1] + 1
            spans.append(Span(start=start, end=end))
        return spans

    def locate(self, quote: str, *, allow_trim: bool = True) -> Located | None:
        """Find every occurrence of ``quote``; None if it cannot be located.

        Tries an exact match, then a normalized one (whitespace, quote marks,
        dashes, emphasis and case), then, if ``allow_trim``, a normalized
        match after trimming wrapping quotes and trailing punctuation.
        """
        stripped = quote.strip()
        if not stripped:
            return None
        exact = [Span(start=s, end=s + len(stripped)) for s in _find_all(self.text, stripped)]
        if exact:
            return Located(exact, MatchMethod.EXACT)
        nquote = normalize(stripped)[0].strip()
        if not nquote:
            return None
        spans = self._normalized_spans(nquote)
        if spans:
            return Located(spans, MatchMethod.NORMALIZED)
        if allow_trim:
            trimmed = nquote.strip(_TRIM_CHARS)
            if trimmed and trimmed != nquote:
                spans = self._normalized_spans(trimmed)
                if spans:
                    return Located(spans, MatchMethod.TRIMMED)
        return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n")
_SENTENCE_RE = re.compile(r"[.!?][\"')\]]*\s")
_SPACE_RE = re.compile(r"\s")


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice ``[start, end)`` of the document."""

    start: int
    end: int
    text: str


def _last_break(text: str, lo: int, hi: int) -> int | None:
    window = text[lo:hi]
    for pattern in (_PARAGRAPH_RE, _SENTENCE_RE, _SPACE_RE):
        cut = None
        for match in pattern.finditer(window):
            cut = match.end()
        if cut is not None and cut > len(window) // 4:
            return lo + cut
    return None


def split_into_chunks(text: str, max_chars: int) -> list[Chunk]:
    """Split ``text`` into gap-free chunks of at most ``max_chars`` characters.

    Cuts prefer paragraph breaks, then sentence ends, then whitespace.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    chunks: list[Chunk] = []
    start = 0
    n = len(text)
    while start < n:
        hi = min(n, start + max_chars)
        end = hi if hi == n else (_last_break(text, start, hi) or hi)
        chunks.append(Chunk(start=start, end=end, text=text[start:end]))
        start = end
    return chunks


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(\S.*?)[ \t#]*$", re.MULTILINE)


@dataclass(frozen=True)
class Heading:
    """A markdown heading: ``level`` 1-6 and the span of the whole line text."""

    level: int
    span: Span
    title: str


def headings(text: str) -> list[Heading]:
    """Markdown ATX headings in document order.

    The span covers the heading line from the first ``#`` to the end of the
    title (trailing whitespace and closing ``#`` excluded).
    """
    out: list[Heading] = []
    for m in _HEADING_RE.finditer(text):
        out.append(
            Heading(
                level=len(m.group(1)),
                span=Span(start=m.start(1), end=m.end(2)),
                title=m.group(2),
            )
        )
    return out


def title_span(text: str) -> Span | None:
    """Span of the document's title: its first heading, else its first line.

    None only for a document with no non-whitespace text.
    """
    found = headings(text)
    if found:
        return found[0].span
    for m in re.finditer(r"[^\n]+", text):
        line = m.group(0)
        stripped = line.strip()
        if stripped:
            start = m.start() + (len(line) - len(line.lstrip()))
            return Span(start=start, end=start + len(stripped))
    return None


# ---------------------------------------------------------------------------
# Untrusted-text delimiting
# ---------------------------------------------------------------------------


def new_nonce() -> str:
    """A random marker id for delimiting untrusted text."""
    return secrets.token_hex(8)


def delimit(label: str, body: str, nonce: str) -> str:
    """Wrap ``body`` in ``<<<BEGIN label nonce>>>`` / ``<<<END label nonce>>>``.

    The random nonce stops text inside ``body`` from closing the block early.
    """
    return f"<<<BEGIN {label} {nonce}>>>\n{body}\n<<<END {label} {nonce}>>>"
