"""Report renderer (T8): ``Report`` to markdown, HTML email and plain text.

Section order follows the design doc: verdict line and counts, must fix,
strongest counter-case, should fix, polish, originality notes (heuristic),
appendix (claim ledger), then a footer with usage and notices. Empty
sections render ``EMPTY_SECTION`` ("None found.") in every format.

All document text, claim text, messages and evidence are treated as
untrusted: escaped for HTML and markdown, and only ``http``/``https`` URLs
are linked.
"""

from reviewdesk.report.email import RenderedEmail, email_subject, render_email
from reviewdesk.report.html import render_html
from reviewdesk.report.markdown import render_markdown
from reviewdesk.report.text import render_text
from reviewdesk.report.view import EMPTY_SECTION, ReportView, build_view, safe_url

__all__ = [
    "EMPTY_SECTION",
    "RenderedEmail",
    "ReportView",
    "build_view",
    "email_subject",
    "render_email",
    "render_html",
    "render_markdown",
    "render_text",
    "safe_url",
]
