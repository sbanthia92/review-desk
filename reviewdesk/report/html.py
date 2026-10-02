"""Render a ``Report`` to an HTML email body.

Uses a Jinja2 template with autoescaping on, so every untrusted string is
HTML-escaped. Styling is inline CSS only: no scripts, stylesheets, images or
other external assets. Only ``http``/``https`` evidence URLs become links.
"""

from __future__ import annotations

from functools import cache

from jinja2 import Environment, PackageLoader, StrictUndefined

from reviewdesk.contracts.models import Document, Report
from reviewdesk.report.view import (
    EMPTY_SECTION,
    ORIGINALITY_DISCLAIMER,
    UNVERIFIED_NOTE,
    ReportView,
    build_view,
)

TEMPLATE_NAME = "email.html.j2"


@cache
def _environment() -> Environment:
    return Environment(
        loader=PackageLoader("reviewdesk.report", "templates"),
        autoescape=True,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


def render_view_html(view: ReportView) -> str:
    """Render an already-built view to HTML."""
    template = _environment().get_template(TEMPLATE_NAME)
    return template.render(
        view=view,
        empty=EMPTY_SECTION,
        disclaimer=ORIGINALITY_DISCLAIMER,
        unverified_note=UNVERIFIED_NOTE,
    )


def render_html(report: Report, document: Document | None = None) -> str:
    """Render ``report`` to a self-contained HTML email body."""
    return render_view_html(build_view(report, document))
