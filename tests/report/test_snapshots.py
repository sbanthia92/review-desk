"""Snapshot tests for the markdown, HTML and plain-text renderers.

Expected outputs live in ``tests/report/snapshots/`` and are compared exactly.
Regenerate them after an intentional change with::

    UPDATE_SNAPSHOTS=1 uv run pytest tests/report

then review the diff before committing.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from reviewdesk.contracts import Document, Report
from reviewdesk.report import render_email, render_html, render_markdown, render_text
from reviewdesk.testing.fakes import OPINION_DOC, SHORT_DOC, empty_report, sample_report

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
UPDATE = os.environ.get("UPDATE_SNAPSHOTS") == "1"

Renderer = Callable[[Report, Document | None], str]

RENDERERS: dict[str, Renderer] = {
    "md": render_markdown,
    "html": render_html,
    "txt": render_text,
}


def sample_report_with_notes() -> Report:
    """``sample_report`` plus footer notices and timing (every footer field set)."""
    report = sample_report()
    return report.model_copy(
        update={
            "notes": ["fact-check unavailable", "partial results: time limit reached"],
            "usage": report.usage.model_copy(update={"fetch_calls": 4, "seconds": 87.25}),
        }
    )


CASES: dict[str, tuple[Callable[[], Report], Document | None]] = {
    "sample_with_document": (sample_report, OPINION_DOC),
    "sample_with_notes": (sample_report_with_notes, OPINION_DOC),
    "sample_without_document": (sample_report, None),
    "empty": (empty_report, SHORT_DOC),
}


def assert_snapshot(name: str, actual: str) -> None:
    path = SNAPSHOT_DIR / name
    if UPDATE or not path.exists():
        if not UPDATE:
            pytest.fail(f"missing snapshot {path.name}; run with UPDATE_SNAPSHOTS=1 to create it")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        return
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{path.name} differs from snapshot; if intended, rerun with UPDATE_SNAPSHOTS=1"
    )


@pytest.mark.parametrize("fmt", sorted(RENDERERS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_snapshot(case: str, fmt: str) -> None:
    make_report, document = CASES[case]
    output = RENDERERS[fmt](make_report(), document)
    assert_snapshot(f"{case}.{fmt}", output)


@pytest.mark.parametrize("case", sorted(CASES))
def test_email_parts_match_snapshots(case: str) -> None:
    make_report, document = CASES[case]
    email = render_email(make_report(), document)
    assert email.html == (SNAPSHOT_DIR / f"{case}.html").read_text(encoding="utf-8")
    assert email.text == (SNAPSHOT_DIR / f"{case}.txt").read_text(encoding="utf-8")


def test_rendering_is_deterministic() -> None:
    for fmt, render in RENDERERS.items():
        assert render(sample_report(), OPINION_DOC) == render(sample_report(), OPINION_DOC), fmt
