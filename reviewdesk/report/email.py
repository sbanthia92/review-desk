"""Build a complete report email: subject, HTML body and plain-text fallback."""

from __future__ import annotations

from dataclasses import dataclass

from reviewdesk.contracts.models import Document, Report
from reviewdesk.report.html import render_view_html
from reviewdesk.report.text import render_view_text
from reviewdesk.report.view import build_view, one_line

SUBJECT_PREFIX = "Review Desk report"
MAX_SUBJECT_CHARS = 120


@dataclass(frozen=True)
class RenderedEmail:
    """A rendered report email. ``subject`` is a single safe header line."""

    subject: str
    html: str
    text: str


def email_subject(report: Report) -> str:
    """Subject line: prefix plus the verdict, one line, no control characters.

    The verdict line is model output, so CR/LF (header injection) and other
    control characters are removed and the result is length-capped.
    """
    verdict = one_line(report.verdict_line)
    subject = f"{SUBJECT_PREFIX}: {verdict}" if verdict else SUBJECT_PREFIX
    return one_line(subject, MAX_SUBJECT_CHARS)


def render_email(report: Report, document: Document | None = None) -> RenderedEmail:
    """Render ``report`` as an email with HTML and plain-text parts."""
    view = build_view(report, document)
    return RenderedEmail(
        subject=email_subject(report),
        html=render_view_html(view),
        text=render_view_text(view),
    )
