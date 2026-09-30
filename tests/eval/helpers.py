"""Shared builders for eval tests: a tiny seeded document and a fake judge."""

from __future__ import annotations

from eval.dataset import SeededDocument
from eval.items import ReportItem
from reviewdesk.contracts import Document, Evidence, Span
from reviewdesk.testing.fakes import FIXED_TIME

BORROWED = "It is not from the benevolence of the butcher that we expect our dinner."

TEXT = (
    "# Pressing wins titles\n\n"
    "Liverpool won the 2019-20 league with 105 points. "
    "Pressing teams concede 50% fewer shots. "
    "If one pressing team won, pressing always wins. "
    f"{BORROWED} "
    "The players was tired. "
    "Their is no debate left, and City scored 90 goals that season. "
    "The club was founded in 1892.\n"
)


def tiny_seeded() -> SeededDocument:
    """A seeded document with one defect of every type and every conflict rule."""
    return SeededDocument.model_validate(
        {
            "id": "tiny",
            "title": "Pressing wins titles",
            "genre": "football_blog",
            "profile": "opinion",
            "text": TEXT,
            "defects": [
                {
                    "id": "wrong",
                    "type": "wrong_fact",
                    "quote": "Liverpool won the 2019-20 league with 105 points.",
                    "correction": "They had 99 points.",
                },
                {
                    "id": "unsup",
                    "type": "unsupported",
                    "quote": "Pressing teams concede 50% fewer shots.",
                },
                {
                    "id": "weak",
                    "type": "weak_argument",
                    "quote": "If one pressing team won, pressing always wins.",
                    "known_rebuttal": "One title is an outlier, not a law.",
                },
                {
                    "id": "borrowed",
                    "type": "borrowed_sentence",
                    "quote": BORROWED,
                    "source_url": "https://www.gutenberg.org/ebooks/3300",
                },
                {"id": "gram", "type": "grammar", "quote": "The players was tired."},
                {
                    "id": "c_copy",
                    "type": "conflict",
                    "rule": "copyedit_yields_to_fact",
                    "quote": "Liverpool won the 2019-20 league with 105 points.",
                },
                {
                    "id": "c_verified",
                    "type": "conflict",
                    "rule": "rebuttal_on_verified_fact",
                    "quote": "The club was founded in 1892.",
                },
                {
                    "id": "c_merge",
                    "type": "conflict",
                    "rule": "same_span_merge",
                    "quote": "Their is no debate left",
                },
                {
                    "id": "c_free",
                    "type": "conflict",
                    "rule": "evidence_free_rebuttal",
                    "quote": "If one pressing team won, pressing always wins.",
                },
            ],
        }
    )


def span(quote: str, text: str = TEXT) -> Span:
    """Span of ``quote`` in ``text``."""
    start = text.index(quote)
    return Span(start=start, end=start + len(quote))


def ev(url: str = "https://example.org/a", excerpt: str = "Liverpool had 99 points.") -> Evidence:
    """An evidence item."""
    return Evidence(url=url, title="Source", excerpt=excerpt, retrieved_at=FIXED_TIME)


class StubJudge:
    """``EvalJudge`` with fixed answers; records calls."""

    def __init__(
        self, *, real: bool | None = True, supports: bool | None = True, score: int | None = 4
    ) -> None:
        self.real = real
        self.supports_answer = supports
        self.score = score
        self.calls: list[str] = []

    async def is_real(self, doc: Document, item: ReportItem) -> bool | None:
        self.calls.append(f"real:{item.id}")
        return self.real

    async def supports(self, statement: str, evidence: Evidence) -> bool | None:
        self.calls.append(f"supports:{evidence.url}")
        return self.supports_answer

    async def rebuttal_strength(
        self, doc: Document, argument: str, known_rebuttal: str, candidate: str
    ) -> int | None:
        self.calls.append("rebuttal")
        return self.score
