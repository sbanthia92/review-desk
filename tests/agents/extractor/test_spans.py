"""Span location: exact, normalized fallback, and unlocatable quotes."""

from __future__ import annotations

import itertools
import re

import pytest

from reviewdesk.agents.extractor.chunks import split_into_chunks
from reviewdesk.agents.extractor.spans import MatchMethod, TextLocator, normalize
from reviewdesk.testing.fakes import REPORT_DOC, sample_documents


def _only(locator: TextLocator, quote: str):
    located = locator.locate(quote)
    assert located is not None, quote
    return located


def test_exact_match_round_trips():
    text = REPORT_DOC.text
    quote = "68% said they prefer hybrid work"
    located = _only(TextLocator(text), quote)
    assert located.method is MatchMethod.EXACT
    (span,) = located.spans
    assert text[span.start : span.end] == quote


def test_exact_match_strips_surrounding_whitespace():
    located = _only(TextLocator("Alpha beta. Gamma delta."), "  Gamma delta. \n")
    assert located.method is MatchMethod.EXACT
    assert located.spans[0].start == 12


@pytest.mark.parametrize(
    ("text", "quote", "expected"),
    [
        # Curly quotes in the model output, straight in the document.
        (
            'He said "offices are for meetings, not work." Then left.',
            "He said “offices are for meetings, not work.”",
            'He said "offices are for meetings, not work."',
        ),
        # Curly apostrophe in the document, straight in the quote.
        ("The company’s founder spoke.", "The company's founder", "The company’s founder"),
        # Line wrap and double spaces in the document.
        (
            "Remote staff\nfiled  12% fewer\ttickets.",
            "Remote staff filed 12% fewer tickets.",
            "Remote staff\nfiled  12% fewer\ttickets.",
        ),
        # Em dash vs hyphen, case difference.
        ("Growth — not decline — is here.", "growth - not decline", "Growth — not decline"),
        # Markdown emphasis in the document that the model dropped.
        ("This is **very** important.", "This is very important.", "This is **very** important."),
        # Non-breaking space.
        ("It costs 5 dollars.", "It costs 5 dollars.", "It costs 5 dollars."),
    ],
)
def test_normalized_fallback_maps_back_to_original(text: str, quote: str, expected: str):
    located = _only(TextLocator(text), quote)
    assert located.method is MatchMethod.NORMALIZED
    (span,) = located.spans
    assert text[span.start : span.end] == expected


def test_trimmed_fallback_ignores_added_punctuation():
    text = "Water boils at 90 degrees Celsius at sea level and nowhere else"
    located = _only(TextLocator(text), '"Water boils at 90 degrees Celsius at sea level."')
    assert located.method is MatchMethod.TRIMMED
    span = located.spans[0]
    assert text[span.start : span.end] == "Water boils at 90 degrees Celsius at sea level"


def test_elided_quote_spans_the_gap():
    text = "Store sessions in a single Redis instance with a 30-day TTL. Next."
    located = _only(TextLocator(text), "Store sessions in a single ... with a 30-day TTL")
    assert located.method is MatchMethod.ELIDED
    span = located.spans[0]
    assert text[span.start : span.end] == (
        "Store sessions in a single Redis instance with a 30-day TTL"
    )


@pytest.mark.parametrize(
    "quote",
    [
        "Remote work is illegal in 14 countries.",
        "",
        "   ",
        "***",
        "Of 1,200 employees ... never appears here",
    ],
)
def test_unlocatable_quotes_return_none(quote: str):
    assert TextLocator(REPORT_DOC.text).locate(quote) is None


def test_repeated_quote_returns_every_occurrence():
    text = "Buy now. Sell later. Buy now."
    located = _only(TextLocator(text), "Buy now.")
    assert [s.start for s in located.spans] == [0, 21]


def test_normalize_index_map_points_at_source_characters():
    text = "  A’b —\n\nC…"
    norm, index = normalize(text)
    assert norm == "a'b - c..."
    assert len(norm) == len(index)
    assert all(0 <= i < len(text) for i in index)
    assert index == sorted(index)


@pytest.mark.parametrize("doc", sample_documents(), ids=lambda d: d.id)
def test_every_sentence_of_samples_round_trips(doc):
    """Every sentence, and a mangled copy of it, locates to an exact slice."""
    locator = TextLocator(doc.text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", doc.text) if len(s.strip()) > 3]
    assert sentences
    for sentence in sentences:
        for quote in (sentence, re.sub(r"\s+", "  ", sentence.upper()).replace('"', "“")):
            located = locator.locate(quote)
            assert located is not None, quote
            for span in located.spans:
                sliced = doc.text[span.start : span.end]
                assert normalize(sliced)[0] == normalize(quote.strip())[0]


def test_chunks_cover_text_exactly():
    text = ("Paragraph one has words. " * 20 + "\n\n") * 10 + "x" * 500
    chunks = split_into_chunks(text, 300)
    assert "".join(c.text for c in chunks) == text
    assert all(len(c.text) <= 300 for c in chunks)
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    for a, b in itertools.pairwise(chunks):
        assert a.end == b.start
        assert text[a.start : a.end] == a.text


def test_chunks_prefer_paragraph_breaks():
    text = "a" * 150 + "\n\n" + "b" * 150
    chunks = split_into_chunks(text, 200)
    assert chunks[0].text == "a" * 150 + "\n\n"


def test_chunks_reject_non_positive_size():
    with pytest.raises(ValueError):
        split_into_chunks("abc", 0)
