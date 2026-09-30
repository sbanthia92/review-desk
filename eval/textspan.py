"""Locate a model-quoted string in the original text.

Models often change whitespace, quote marks, dashes or case when quoting. The
locator tries an exact match first, then a normalised match whose offsets are
mapped back to the original text, so the returned span always slices the
original.
"""

from __future__ import annotations

from reviewdesk.contracts import Span

_CHAR_MAP = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    " ": " ",
}


def _normalise(text: str) -> tuple[str, list[int]]:
    """Lower-case, map typographic characters and collapse whitespace.

    Returns the normalised string and, for each of its characters, the index
    of the original character it came from.
    """
    out: list[str] = []
    index: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        ch = _CHAR_MAP.get(ch, ch)
        if ch.isspace():
            if prev_space or not out:
                continue
            out.append(" ")
            index.append(i)
            prev_space = True
            continue
        prev_space = False
        out.append(ch.lower())
        index.append(i)
    if out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def locate_quote(text: str, quote: str) -> Span | None:
    """Span of the first occurrence of ``quote`` in ``text``, or None."""
    quote = quote.strip()
    if not quote:
        return None
    start = text.find(quote)
    if start != -1:
        return Span(start=start, end=start + len(quote))
    norm_text, index = _normalise(text)
    norm_quote, _ = _normalise(quote)
    for candidate in (norm_quote, norm_quote.strip(" .\"'")):
        if not candidate:
            continue
        pos = norm_text.find(candidate)
        if pos != -1:
            return Span(start=index[pos], end=index[pos + len(candidate) - 1] + 1)
    return None
