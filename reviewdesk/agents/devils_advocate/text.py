"""Deterministic text helpers: find-in-page, excerpt verification, domains.

Nothing here calls a model. ``find_in_page`` keeps prompts small by passing
only the passages of a fetched page that match the claim; ``locate_excerpt``
makes sure every quoted excerpt really comes from a retrieved source, so the
model cannot invent citations.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")

_STOPWORDS = frozenset(
    """
    a about above after again against all also an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers him his how i if in into is
    it its itself just me more most my no nor not now of off on once only or other our
    out over own same she should so some such than that the their them then there these
    they this those through to too under until up very was we were what when where which
    while who whom why will with would you your
    """.split()  # noqa: SIM905
)

MAX_EXCERPT_CHARS = 500
PASSAGE_CHARS = 600
MAX_PASSAGES = 3
SHORT_PAGE_CHARS = 1_500


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return " ".join(text.lower().split())


def keywords(*texts: str) -> set[str]:
    """Content words (length ≥ 3, not stopwords) from ``texts``."""
    words: set[str] = set()
    for text in texts:
        for word in _WORD_RE.findall(text.lower()):
            if len(word) >= 3 and word not in _STOPWORDS:
                words.add(word)
    return words


def split_passages(text: str) -> list[str]:
    """Split text into sentence-ish passages."""
    return [p.strip() for p in _SENTENCE_RE.split(text) if p and p.strip()]


def find_in_page(text: str, terms: set[str], *, max_passages: int = MAX_PASSAGES) -> list[str]:
    """Return up to ``max_passages`` windows of ``text`` richest in ``terms``.

    Short pages are returned whole. Windows are built from consecutive
    sentences up to ``PASSAGE_CHARS`` and returned in document order.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= SHORT_PAGE_CHARS:
        return [text]
    sentences = split_passages(text)
    windows: list[tuple[int, int, str]] = []  # (first sentence, last + 1, text)
    for i in range(len(sentences)):
        window = sentences[i]
        j = i + 1
        while j < len(sentences) and len(window) + len(sentences[j]) + 1 <= PASSAGE_CHARS:
            window = f"{window} {sentences[j]}"
            j += 1
        windows.append((i, j, window[:PASSAGE_CHARS]))
    scored = sorted(windows, key=lambda w: (-len(keywords(w[2]) & terms), w[0]))
    chosen: list[tuple[int, int, str]] = []
    for start, end, window in scored:
        if len(chosen) >= max_passages or not keywords(window) & terms:
            break
        if any(start < c_end and c_start < end for c_start, c_end, _ in chosen):
            continue  # overlaps a passage already chosen
        chosen.append((start, end, window))
    if not chosen:
        return [text[:PASSAGE_CHARS]]
    return [w for _, _, w in sorted(chosen)]


def locate_excerpt(excerpt: str, source_text: str) -> str | None:
    """Return a verified excerpt from ``source_text``, or None.

    An exact (case- and whitespace-insensitive) match returns the excerpt with
    normalized whitespace. Otherwise the source sentence sharing at least 70%
    of the excerpt's content words is returned, so light paraphrase is repaired
    to the real quote. Anything else is treated as invented and rejected.
    """
    cleaned = " ".join(excerpt.split())
    if len(cleaned) < 8:
        return None
    if normalize(cleaned) in normalize(source_text):
        return cleaned[:MAX_EXCERPT_CHARS]
    wanted = keywords(cleaned)
    if len(wanted) < 3:
        return None
    best: tuple[float, str] | None = None
    for sentence in split_passages(source_text):
        overlap = len(wanted & keywords(sentence)) / len(wanted)
        if best is None or overlap > best[0]:
            best = (overlap, sentence)
    if best is not None and best[0] >= 0.7:
        return " ".join(best[1].split())[:MAX_EXCERPT_CHARS]
    return None


def domain(url: str) -> str:
    """Registrable-ish host of ``url`` (``www.`` stripped), lowercased."""
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


def urls_in(text: str) -> set[str]:
    """HTTP(S) URLs mentioned in ``text``, trailing punctuation stripped."""
    return {u.rstrip(".,;:") for u in _URL_RE.findall(text)}
