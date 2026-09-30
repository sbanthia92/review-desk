"""Split long documents into contiguous chunks for extraction.

Chunks cover the whole text with no gaps or overlap, so a span found inside a
chunk maps to the document by adding ``chunk.start``. Splits prefer paragraph
breaks, then sentence ends, then any whitespace, and only cut mid-word when a
chunk has no whitespace at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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
    """Best cut point in ``text[lo:hi]`` (an offset just after a break)."""
    window = text[lo:hi]
    for pattern in (_PARAGRAPH_RE, _SENTENCE_RE, _SPACE_RE):
        cut = None
        for match in pattern.finditer(window):
            cut = match.end()
        # Ignore breaks in the first quarter so chunks do not get tiny.
        if cut is not None and cut > len(window) // 4:
            return lo + cut
    return None


def split_into_chunks(text: str, max_chars: int) -> list[Chunk]:
    """Split ``text`` into chunks of at most ``max_chars`` characters."""
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
