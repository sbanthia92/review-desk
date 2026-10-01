"""Deterministic sampling of distinctive sentences.

The document is split into prose sentences with exact character spans
(``doc.text[span.start:span.end] == sentence.text``). Headings, code blocks,
block quotes, table rows, sentences that mostly quote someone else and common
boilerplate are skipped. The rest are scored for distinctiveness (long and
heavy in uncommon words) and ranked; ties break by position, so the same
document always yields the same sample.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from reviewdesk.contracts.models import Span

MIN_WORDS = 8
"""Sentences with fewer words are too short to be distinctive."""

MIN_UNCOMMON_WORDS = 4
"""Minimum distinct uncommon words for a sentence to qualify."""

MIN_UNCOMMON_RATIO = 0.3
"""Minimum share of uncommon words among all words."""

MAX_QUOTED_RATIO = 0.5
"""Sentences with more than this share of characters inside quotes are skipped."""

_COMMON_WORDS_TEXT = """
    a about above after again against all also although always am an and any are
    around as at away back be because been before being below between both but by
    can could did do does doing done down during each either else even ever every
    few first for from further get gets getting go goes going good got great had has
    have having he her here hers herself him himself his how however i if in into is
    it its itself just know last less like little long made make makes many may me
    might more most much must my myself never new next no nor not now of off often
    on once one only or other others our ours ourselves out over own people really
    right same say says said see seem seems she should since so some something still
    such take than that the their theirs them themselves then there these they thing
    things think this those though through thus time to too two under until up upon
    us use used very want was way we well were what when where whether which while
    who whom whose why will with within without would yes yet you your yours
    yourself yourselves year years day days today
"""
COMMON_WORDS: frozenset[str] = frozenset(_COMMON_WORDS_TEXT.split())
"""Very common English words; everything else counts as uncommon."""

_BOILERPLATE = re.compile(
    r"all rights reserved|copyright|\(c\)\s*\d{4}|©|subscribe|sign up|newsletter|"
    r"click here|read more|table of contents|cookie|privacy policy|terms of (?:use|service)|"
    r"share this|follow us|posted (?:on|by)|last updated|this article (?:was|has been)",
    re.IGNORECASE,
)

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|$)")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+")
_TABLE_OR_RULE = re.compile(r"^\s*(?:\||[-=*_]{3,}\s*$)")
_ABBREVIATIONS_TEXT = (
    "mr mrs ms dr prof sr jr st vs etc e.g i.e cf fig no inc ltd co corp approx dept est"
)
_ABBREVIATIONS = frozenset(_ABBREVIATIONS_TEXT.split())
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”")


@dataclass(frozen=True)
class Sentence:
    """A candidate sentence with its exact span and distinctiveness score."""

    span: Span
    text: str
    word_count: int
    uncommon: int
    score: float


def words(text: str) -> list[str]:
    """Lower-cased word tokens of ``text``."""
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def is_uncommon(word: str) -> bool:
    """True for words that are not very common (numbers always count)."""
    if any(ch.isdigit() for ch in word):
        return True
    return len(word) >= 4 and word.replace("’", "'") not in COMMON_WORDS


def _prose_blocks(text: str) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` of prose paragraphs, skipping non-prose lines.

    Headings, fenced code, indented code, block quotes, table rows and
    horizontal rules end the current paragraph and are excluded.
    """
    block_start: int | None = None
    block_end = 0
    in_fence = False
    pos = 0
    for line in text.splitlines(keepends=True):
        line_start, pos = pos, pos + len(line)
        body = line.rstrip("\r\n")
        stripped = body.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            skip = True
        else:
            skip = (
                in_fence
                or not stripped
                or stripped.startswith(("#", ">"))
                or body.startswith(("    ", "\t"))
                or bool(_TABLE_OR_RULE.match(body))
            )
        if skip:
            if block_start is not None:
                yield block_start, block_end
                block_start = None
            continue
        if _BULLET.match(body) and block_start is not None:
            # Each list item is its own paragraph.
            yield block_start, block_end
            block_start = None
        if block_start is None:
            block_start = line_start
        block_end = line_start + len(body)
    if block_start is not None:
        yield block_start, block_end


def _is_boundary(block: str, match: re.Match[str]) -> bool:
    before = block[: match.start()]
    last = before.rsplit(None, 1)[-1] if before.split() else ""
    if match.group(0).startswith(".") and last.lower().rstrip(".") in _ABBREVIATIONS:
        return False
    if len(last) == 1 and last.isupper() and match.group(0) == ".":
        return False  # an initial, e.g. "J. Smith"
    rest = block[match.end() :].lstrip()
    return not rest or not rest[0].islower()


def split_sentences(text: str) -> list[tuple[Span, str]]:
    """Split the prose of ``text`` into sentences with exact spans.

    Leading list markers and surrounding whitespace are excluded from spans.
    """
    out: list[tuple[Span, str]] = []
    for b_start, b_end in _prose_blocks(text):
        block = text[b_start:b_end]
        pieces: list[tuple[int, int]] = []
        prev = 0
        for match in _SENTENCE_END.finditer(block):
            if _is_boundary(block, match):
                pieces.append((prev, match.end()))
                prev = match.end()
        if prev < len(block):
            pieces.append((prev, len(block)))
        for p_start, p_end in pieces:
            piece = block[p_start:p_end]
            bullet = _BULLET.match(piece)
            lead = bullet.end() if bullet else len(piece) - len(piece.lstrip())
            trimmed = piece[lead:].rstrip()
            if not trimmed:
                continue
            start = b_start + p_start + lead
            span = Span(start=start, end=start + len(trimmed))
            out.append((span, trimmed))
    return out


def quoted_ratio(text: str) -> float:
    """Share of characters of ``text`` inside double quotes."""
    if not text:
        return 0.0
    return sum(len(m.group(0)) for m in _QUOTED.finditer(text)) / len(text)


def score_sentence(span: Span, text: str) -> Sentence | None:
    """Score one sentence, or None if it is not a useful sample."""
    tokens = words(text)
    if len(tokens) < MIN_WORDS:
        return None
    if quoted_ratio(text) > MAX_QUOTED_RATIO or _BOILERPLATE.search(text):
        return None
    uncommon = {t for t in tokens if is_uncommon(t)}
    ratio = sum(1 for t in tokens if is_uncommon(t)) / len(tokens)
    if len(uncommon) < MIN_UNCOMMON_WORDS or ratio < MIN_UNCOMMON_RATIO:
        return None
    # Longer sentences with more distinct uncommon words are more
    # fingerprint-like; cap length so run-ons do not dominate.
    score = len(uncommon) * (0.5 + ratio) + min(len(tokens), 40) * 0.1
    return Sentence(
        span=span, text=text, word_count=len(tokens), uncommon=len(uncommon), score=score
    )


def distinctive_sentences(text: str) -> list[Sentence]:
    """All qualifying sentences, most distinctive first (ties by position).

    Sentences with identical wording are kept once (the first occurrence).
    """
    seen: set[tuple[str, ...]] = set()
    ranked: list[Sentence] = []
    for span, sentence in split_sentences(text):
        scored = score_sentence(span, sentence)
        if scored is None:
            continue
        key = tuple(words(sentence))
        if key in seen:
            continue
        seen.add(key)
        ranked.append(scored)
    ranked.sort(key=lambda s: (-s.score, s.span.start))
    return ranked


# ---------------------------------------------------------------------------
# Off-topic sentences
# ---------------------------------------------------------------------------

OFF_TOPIC_MIN_WORDS = 6
"""Shortest sentence considered as an off-topic candidate."""

OFF_TOPIC_MIN_UNCOMMON = 2
"""Minimum distinct uncommon words for an off-topic candidate."""

OFF_TOPIC_MAX_OVERLAP = 0.5
"""Candidates sharing more than this share of topic words with the rest of the
document are on-topic and not sampled as off-topic."""


def _stem(word: str) -> str:
    """Crude stem so ``battles``/``battle`` and ``wins``/``winning`` match."""
    word = word.replace("’", "'").split("'")[0]
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _topic_stems(text: str) -> set[str]:
    return {_stem(t) for t in words(text) if is_uncommon(t) and not t[0].isdigit()}


def off_topic_sentences(text: str) -> list[Sentence]:
    """Sentences that share few topic words with the rest of the document.

    A borrowed line (a proverb, a famous quotation, a passage lifted from
    elsewhere) often talks about something other than the article around it
    and is too short or too plainly worded to rank as "distinctive". These are
    ranked least-related first (ties: longer first, then position). ``score``
    is ``1 - overlap``.
    """
    sentences = [
        (span, sentence, _topic_stems(sentence)) for span, sentence in split_sentences(text)
    ]
    counts: dict[str, int] = {}
    for _span, _sentence, stems in sentences:
        for stem in stems:
            counts[stem] = counts.get(stem, 0) + 1
    seen: set[tuple[str, ...]] = set()
    ranked: list[Sentence] = []
    for span, sentence, stems in sentences:
        tokens = words(sentence)
        if len(tokens) < OFF_TOPIC_MIN_WORDS or len(stems) < OFF_TOPIC_MIN_UNCOMMON:
            continue
        if quoted_ratio(sentence) > MAX_QUOTED_RATIO or _BOILERPLATE.search(sentence):
            continue
        key = tuple(tokens)
        if key in seen:
            continue
        seen.add(key)
        shared = sum(1 for stem in stems if counts[stem] > 1)
        overlap = shared / len(stems)
        if overlap > OFF_TOPIC_MAX_OVERLAP:
            continue
        ranked.append(
            Sentence(
                span=span,
                text=sentence,
                word_count=len(tokens),
                uncommon=len(stems),
                score=1.0 - overlap,
            )
        )
    ranked.sort(key=lambda s: (-s.score, -s.word_count, s.span.start))
    return ranked


def sample_sentences(text: str, limit: int) -> list[Sentence]:
    """Up to ``limit`` sentences to check, alternating the two rankings.

    Takes the most distinctive sentence, then the least on-topic, and so on,
    skipping repeats, so both a lifted aphorism and a lifted dense passage
    are likely to be sampled. Deterministic for a given text.
    """
    pools = [distinctive_sentences(text), off_topic_sentences(text)]
    chosen: list[Sentence] = []
    taken: set[Span] = set()
    indexes = [0, 0]
    while len(chosen) < limit and any(indexes[i] < len(pools[i]) for i in (0, 1)):
        for i in (0, 1):
            while indexes[i] < len(pools[i]) and pools[i][indexes[i]].span in taken:
                indexes[i] += 1
            if indexes[i] < len(pools[i]) and len(chosen) < limit:
                sentence = pools[i][indexes[i]]
                indexes[i] += 1
                taken.add(sentence.span)
                chosen.append(sentence)
    return chosen
