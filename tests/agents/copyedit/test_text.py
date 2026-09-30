"""Tests for the shared text helpers (locator, chunks, headings)."""

from __future__ import annotations

import itertools

import pytest

from reviewdesk.agents.copyedit.text import (
    MatchMethod,
    TextLocator,
    delimit,
    headings,
    split_into_chunks,
    title_span,
)
from reviewdesk.contracts import Span
from reviewdesk.testing.fakes import DESIGN_DOC, OPINION_DOC, SHORT_DOC


def test_exact_and_repeated_matches() -> None:
    loc = TextLocator("a cat and a cat")
    found = loc.locate("a cat")
    assert found is not None
    assert found.method is MatchMethod.EXACT
    assert found.spans == [Span(start=0, end=5), Span(start=10, end=15)]


def test_normalized_match_maps_back_to_original() -> None:
    text = "He said “no”  and\nleft."
    loc = TextLocator(text)
    found = loc.locate('said "no" and left')
    assert found is not None
    assert found.method is MatchMethod.NORMALIZED
    assert found.spans[0].text_of(text) == "said “no”  and\nleft"


def test_trim_is_optional() -> None:
    loc = TextLocator("The data is clear")
    assert loc.locate('"The data is clear."', allow_trim=False) is None
    found = loc.locate('"The data is clear."')
    assert found is not None
    assert found.method is MatchMethod.TRIMMED


def test_unlocatable_and_empty() -> None:
    loc = TextLocator(OPINION_DOC.text)
    assert loc.locate("not in the document at all") is None
    assert loc.locate("   ") is None
    assert loc.locate("**") is None


def test_chunks_cover_text_without_gaps() -> None:
    text = OPINION_DOC.text * 3
    chunks = split_into_chunks(text, 200)
    assert "".join(c.text for c in chunks) == text
    assert all(len(c.text) <= 200 for c in chunks)
    for a, b in itertools.pairwise(chunks):
        assert a.end == b.start
    assert split_into_chunks("x" * 10, 4)[0].text == "xxxx"
    with pytest.raises(ValueError):
        split_into_chunks(text, 0)


def test_headings_and_title() -> None:
    found = headings(DESIGN_DOC.text)
    assert [h.title for h in found] == [
        "Design: Move sessions to Redis",
        "Context",
        "Proposal",
        "Rollout",
    ]
    assert [h.level for h in found] == [1, 2, 2, 2]
    assert found[3].span.text_of(DESIGN_DOC.text) == "## Rollout"
    title = title_span(DESIGN_DOC.text)
    assert title is not None
    assert title.text_of(DESIGN_DOC.text) == "# Design: Move sessions to Redis"
    short = title_span(SHORT_DOC.text)
    assert short is not None
    assert short.text_of(SHORT_DOC.text) == SHORT_DOC.text
    assert title_span("  \n\n  ") is None
    text = "\n   First line  \nsecond"
    indented = title_span(text)
    assert indented is not None
    assert indented.text_of(text) == "First line"


def test_delimit() -> None:
    assert delimit("DOCUMENT", "body", "abc") == (
        "<<<BEGIN DOCUMENT abc>>>\nbody\n<<<END DOCUMENT abc>>>"
    )
