"""Locate LLM-quoted text in the original document.

LLMs are unreliable at character offsets, so the extractor asks for verbatim
quotes and finds them here. Matching is tried in order:

1. ``exact``: the (whitespace-stripped) quote occurs verbatim.
2. ``normalized``: both sides are normalized (whitespace collapsed, curly
   quotes and dashes straightened, markdown emphasis markers and zero-width
   characters dropped, lower-cased) and matched; offsets are mapped back to
   the original text through an index map.
3. ``trimmed``: as 2, after trimming wrapping quote marks and trailing
   punctuation the model may have added.
4. ``elided``: a quote with an ellipsis (``...``) is matched as its first and
   last fragments, and the span covers everything between them.

Every returned span covers a slice of the original text, so callers take the
claim text from ``text[span.start:span.end]`` and it round-trips exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from reviewdesk.contracts.models import Span

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
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.+|…|\[\.\.\.\]|\[…\])\s*")
_MAX_ELISION_CHARS = 2_000
"""Longest gap an elided quote may span."""


class MatchMethod(StrEnum):
    """How a quote was located."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    TRIMMED = "trimmed"
    ELIDED = "elided"


@dataclass(frozen=True)
class Located:
    """Every occurrence of a quote, in document order, and how it was found."""

    spans: list[Span]
    method: MatchMethod


def normalize(text: str) -> tuple[str, list[int]]:
    """Normalize ``text`` for fuzzy matching.

    Returns the normalized string and, for each of its characters, the index
    of the original character it came from. Whitespace runs collapse to one
    space; leading whitespace is dropped.
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


def _normalized_quote(quote: str) -> str:
    return normalize(quote)[0].strip()


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

    def _span_from_norm(self, nstart: int, nend: int) -> Span:
        start = self._index[nstart]
        end = self._index[nend - 1] + 1
        return Span(start=start, end=end)

    def _normalized_spans(self, nquote: str) -> list[Span]:
        return [self._span_from_norm(s, s + len(nquote)) for s in _find_all(self._norm, nquote)]

    def _elided_spans(self, quote: str) -> list[Span]:
        parts = [_normalized_quote(p) for p in _ELLIPSIS_RE.split(quote)]
        parts = [p.strip(_TRIM_CHARS) for p in parts]
        parts = [p for p in parts if p]
        if len(parts) < 2:
            return []
        head, tail = parts[0], parts[-1]
        spans: list[Span] = []
        for hs in _find_all(self._norm, head):
            ts = self._norm.find(tail, hs + len(head))
            if ts == -1 or ts - (hs + len(head)) > _MAX_ELISION_CHARS:
                continue
            spans.append(self._span_from_norm(hs, ts + len(tail)))
        return spans

    def locate(self, quote: str) -> Located | None:
        """Find every occurrence of ``quote``; None if it cannot be located."""
        stripped = quote.strip()
        if not stripped:
            return None
        exact = [Span(start=s, end=s + len(stripped)) for s in _find_all(self.text, stripped)]
        if exact:
            return Located(exact, MatchMethod.EXACT)
        nquote = _normalized_quote(stripped)
        if not nquote:
            return None
        spans = self._normalized_spans(nquote)
        if spans:
            return Located(spans, MatchMethod.NORMALIZED)
        trimmed = nquote.strip(_TRIM_CHARS)
        if trimmed and trimmed != nquote:
            spans = self._normalized_spans(trimmed)
            if spans:
                return Located(spans, MatchMethod.TRIMMED)
        if _ELLIPSIS_RE.search(stripped):
            spans = self._elided_spans(stripped)
            if spans:
                return Located(spans, MatchMethod.ELIDED)
        return None
