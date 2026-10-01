"""Sentence splitting and distinctive-sentence sampling."""

from __future__ import annotations

import pytest

from reviewdesk.agents.originality import distinctive_sentences, split_sentences
from reviewdesk.contracts import Document
from reviewdesk.testing.fakes import OPINION_DOC, sample_documents


@pytest.mark.parametrize("doc", sample_documents(), ids=lambda d: d.id)
def test_spans_round_trip_on_samples(doc: Document) -> None:
    for span, text in split_sentences(doc.text):
        assert span.is_valid_for(doc.text)
        assert doc.text[span.start : span.end] == text
        assert text == text.strip()
    for sentence in distinctive_sentences(doc.text):
        assert sentence.span.text_of(doc.text) == sentence.text


def test_headings_are_never_sampled() -> None:
    texts = [s.text for s in distinctive_sentences(OPINION_DOC.text)]
    assert texts
    assert not any(t.startswith("#") or "Why high pressing wins titles" in t for t in texts)


def test_most_distinctive_first_and_deterministic() -> None:
    first = distinctive_sentences(OPINION_DOC.text)
    second = distinctive_sentences(OPINION_DOC.text)
    assert first == second
    assert first[0].text.startswith("Liverpool won the 2019-20 Premier League")
    scores = [s.score for s in first]
    assert scores == sorted(scores, reverse=True)


def test_short_sentences_are_skipped() -> None:
    texts = [s.text for s in distinctive_sentences(OPINION_DOC.text)]
    assert "Critics say pressing exhausts players by February." not in texts


def test_non_prose_lines_are_skipped() -> None:
    text = (
        "# Heading about quantum entanglement experiments in orbital laboratories\n"
        "\n"
        "> Block quoted: orbital laboratories measure quantum entanglement across continents.\n"
        "\n"
        "```\n"
        "Code comment: orbital laboratories measure quantum entanglement across continents.\n"
        "```\n"
        "\n"
        "    Indented code: orbital laboratories measure entanglement across continents.\n"
        "| Table cell: orbital laboratories measure quantum entanglement | x |\n"
        "\n"
        "Engineers calibrated sapphire resonators before measuring gravitational redshift.\n"
    )
    texts = [s.text for s in distinctive_sentences(text)]
    assert texts == [
        "Engineers calibrated sapphire resonators before measuring gravitational redshift."
    ]


def test_quotes_of_others_and_boilerplate_are_skipped() -> None:
    text = (
        '"Orbital laboratories will measure quantum entanglement across entire continents."\n\n'
        "Copyright 2026 Orbital Laboratories Incorporated, all rights reserved worldwide.\n\n"
        "Subscribe to our newsletter for weekly orbital laboratory entanglement updates.\n\n"
        "Engineers calibrated sapphire resonators before measuring gravitational redshift.\n"
    )
    texts = [s.text for s in distinctive_sentences(text)]
    assert texts == [
        "Engineers calibrated sapphire resonators before measuring gravitational redshift."
    ]


def test_bullets_are_stripped_from_spans() -> None:
    text = (
        "- Engineers calibrated sapphire resonators before measuring gravitational redshift.\n"
        "- Technicians replaced cryogenic valves after detecting microscopic helium leaks.\n"
    )
    found = split_sentences(text)
    assert [t for _, t in found] == [
        "Engineers calibrated sapphire resonators before measuring gravitational redshift.",
        "Technicians replaced cryogenic valves after detecting microscopic helium leaks.",
    ]
    for span, sentence in found:
        assert text[span.start : span.end] == sentence


def test_abbreviations_and_initials_do_not_split() -> None:
    text = (
        "Dr. Okafor and J. Smith measured glacier retreat, e.g. Thwaites, using radar. "
        "Their survey covered forty kilometres of unstable Antarctic coastline."
    )
    assert [t for _, t in split_sentences(text)] == [
        "Dr. Okafor and J. Smith measured glacier retreat, e.g. Thwaites, using radar.",
        "Their survey covered forty kilometres of unstable Antarctic coastline.",
    ]


def test_duplicate_sentences_sampled_once() -> None:
    line = "Engineers calibrated sapphire resonators before measuring gravitational redshift."
    text = f"{line}\n\n{line}\n"
    sampled = distinctive_sentences(text)
    assert len(sampled) == 1
    assert sampled[0].span.start == 0


def test_empty_and_heading_only_documents() -> None:
    assert distinctive_sentences("") == []
    assert (
        distinctive_sentences("# Only a heading with several distinctive technical words\n") == []
    )


def test_off_topic_aphorism_is_sampled():
    from reviewdesk.agents.originality.sentences import off_topic_sentences, sample_sentences

    text = (
        "Leicester City won the Premier League title with a modest wage bill. "
        "Their wage bill was the fourth lowest in the league that season. "
        "The league title usually goes to a club with a huge wage bill. "
        "If you know the enemy and know yourself, you need not fear the result of a "
        "hundred battles. "
        "Leicester City showed that a modest club can still win the league title."
    )
    borrowed = "If you know the enemy and know yourself"
    assert off_topic_sentences(text)[0].text.startswith(borrowed)
    sample = sample_sentences(text, 2)
    assert any(s.text.startswith(borrowed) for s in sample)
    assert len({s.span for s in sample}) == len(sample)  # no repeats
    for s in sample:
        assert s.span.text_of(text) == s.text
    assert sample_sentences(text, 0) == []
    assert sample_sentences(text, 3) == sample_sentences(text, 3)  # deterministic
