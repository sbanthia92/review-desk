"""Near-match scoring and URL comparison."""

from __future__ import annotations

from reviewdesk.agents.originality import DEFAULT_THRESHOLD, normalize_url, same_url, similarity
from reviewdesk.agents.originality.similarity import MAX_EXCERPT_CHARS

SENTENCE = "Engineers calibrated sapphire resonators before measuring gravitational redshift."


def test_verbatim_scores_one_with_excerpt() -> None:
    other = f"Lab news. {SENTENCE.upper()} More text follows here."
    match = similarity(SENTENCE, other)
    assert match.score == 1.0
    assert match.excerpt.lower() == SENTENCE.lower().rstrip(".")


def test_light_edit_scores_high() -> None:
    other = "Engineers calibrated the sapphire resonators before measuring gravitational redshift"
    assert similarity(SENTENCE, other).score >= DEFAULT_THRESHOLD


def test_same_topic_different_wording_scores_low() -> None:
    other = "Gravitational redshift was measured after the team tuned its sapphire devices."
    assert similarity(SENTENCE, other).score < 0.3


def test_unrelated_and_empty() -> None:
    assert similarity(SENTENCE, "Bread recipes for beginners.").score == 0.0
    assert similarity(SENTENCE, "").score == 0.0
    assert similarity("", "anything").score == 0.0


def test_long_page_uses_best_window() -> None:
    filler = "Unrelated words about gardening and tomatoes grow here. " * 200
    page = filler + SENTENCE + " " + filler
    match = similarity(SENTENCE, page)
    assert match.score == 1.0
    assert "sapphire resonators" in match.excerpt
    assert len(match.excerpt) <= MAX_EXCERPT_CHARS


def test_excerpt_is_truncated() -> None:
    long_sentence = " ".join(f"word{i}" for i in range(120))
    match = similarity(long_sentence, long_sentence)
    assert match.score == 1.0
    assert len(match.excerpt) <= MAX_EXCERPT_CHARS
    assert match.excerpt.endswith("…")


def test_url_normalization() -> None:
    assert normalize_url("https://www.Example.com/post/") == "example.com/post"
    assert same_url("http://example.com/post#top", "https://www.example.com/post/")
    assert not same_url("https://example.com/post?id=1", "https://example.com/post?id=2")
    assert not same_url("https://example.com/a", None)
    assert not same_url(None, None)
