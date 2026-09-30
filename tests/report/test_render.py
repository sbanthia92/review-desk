"""Behavior tests for the report renderers: order, empties, quoting, safety."""

from __future__ import annotations

import re

import pytest

from reviewdesk.contracts import (
    AgentName,
    Claim,
    ClaimLedger,
    ClaimType,
    Document,
    Evidence,
    Finding,
    Profile,
    Rebuttal,
    Report,
    Severity,
    Span,
    Usage,
)
from reviewdesk.report import (
    EMPTY_SECTION,
    build_view,
    email_subject,
    render_email,
    render_html,
    render_markdown,
    render_text,
    safe_url,
)
from reviewdesk.report.markdown import md
from reviewdesk.report.view import MAX_QUOTE_CHARS, clean, one_line
from reviewdesk.testing.fakes import (
    FIXED_TIME,
    OPINION_DOC,
    SHORT_DOC,
    empty_report,
    sample_report,
    span_of,
)

SECTIONS = [
    "Must fix",
    "Strongest counter-case",
    "Should fix",
    "Polish",
    "Originality notes (heuristic)",
    "Appendix: claim ledger",
]

ALL_RENDERERS = [render_markdown, render_html, render_text]


def _hostile_report() -> tuple[Report, Document]:
    text = (
        "Intro paragraph with [a link](javascript:alert(1)) and ![img](http://x/y.png).\n\n"
        "# not a heading <b>bold</b> <script>alert('doc')</script> & more.\n"
    )
    doc = Document.from_text(text, id="doc_hostile")
    quote_span = span_of(text, "[a link](javascript:alert(1)) and ![img](http://x/y.png)")
    tag_span = span_of(text, "# not a heading <b>bold</b> <script>alert('doc')</script>")
    ledger = ClaimLedger()
    ledger.add_claim(
        Claim(
            id="claim_x",
            text=tag_span.text_of(text),
            span=tag_span,
            type=ClaimType.FACTUAL,
            importance=0.5,
        )
    )
    bad_evidence = Evidence(
        url="javascript:alert(document.cookie)",
        title="<img src=x onerror=alert(1)>",
        excerpt="](http://evil.example) <script>alert('ev')</script>",
        retrieved_at=FIXED_TIME,
    )
    quote_url = Evidence(
        url='https://example.org/a"onmouseover="alert(1)',
        title="Quoted URL",
        excerpt="",
        retrieved_at=FIXED_TIME,
    )
    finding = Finding(
        id="find_x",
        agent=AgentName.FACTCHECK,
        severity=Severity.FACTUAL_ERROR,
        span=quote_span,
        message="Bad <script>alert('msg')</script>\n# Injected heading\n[click](http://evil.example)",
        evidence=[bad_evidence, quote_url],
        suggestion="Use **bold** _here_",
    )
    rebuttal = Rebuttal(
        id="reb_x",
        target_claim_ids=["claim_x", "claim_missing"],
        argument="<iframe src='http://evil.example'></iframe>",
        evidence=[bad_evidence],
        strength=5,
    )
    report = Report(
        document_id=doc.id,
        profile=Profile.OPINION,
        verdict_line="Not ready\r\nBcc: victim@example.com <script>x</script>",
        counts={"factual_error": 1},
        must_fix=[finding],
        counter_case=[rebuttal],
        ledger=ledger,
        notes=["fact-check unavailable <script>n</script>"],
        generated_at=FIXED_TIME,
    )
    return report, doc


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("render", ALL_RENDERERS)
def test_sections_in_design_doc_order(render) -> None:
    out = render(sample_report(), OPINION_DOC)
    positions = [out.index("Verdict:")] + [out.index(s) for s in SECTIONS] + [out.index("Usage:")]
    assert positions == sorted(positions)


@pytest.mark.parametrize("render", ALL_RENDERERS)
def test_empty_report_renders_none_found_for_every_section(render) -> None:
    out = render(empty_report(), SHORT_DOC)
    for section in SECTIONS:
        assert section in out
    assert out.count(EMPTY_SECTION) == len(SECTIONS)
    assert "Findings by severity:" in out


@pytest.mark.parametrize("render", ALL_RENDERERS)
def test_populated_report_has_no_empty_markers(render) -> None:
    assert EMPTY_SECTION not in render(sample_report(), OPINION_DOC)


@pytest.mark.parametrize("render", ALL_RENDERERS)
def test_footer_has_usage_and_notes(render) -> None:
    report = sample_report().model_copy(
        update={
            "notes": ["fact-check unavailable", "partial results: time limit reached"],
            "usage": Usage(input_tokens=1_234_567, output_tokens=1, search_calls=7, seconds=12.34),
        }
    )
    out = render(report, OPINION_DOC)
    footer = out[out.index("Usage:") :]
    assert "1,234,568 tokens" in footer
    assert "7 search calls" in footer
    assert "12.3s" in footer
    assert "fact-check unavailable" in footer
    assert "partial results: time limit reached" in footer


def test_counts_ordered_by_severity_and_zeros_hidden() -> None:
    report = empty_report().model_copy(
        update={"counts": {"style": 2, "factual_error": 1, "structure": 0, "custom": 3}}
    )
    view = build_view(report)
    assert [(c.label, c.count) for c in view.counts] == [
        ("factual error", 1),
        ("style", 2),
        ("custom", 3),
    ]


def test_counter_case_keeps_top_three_by_strength() -> None:
    base = sample_report()
    rebuttals = [
        Rebuttal(id=f"reb_{s}", target_claim_ids=["claim_thesis"], argument=f"arg {s}", strength=s)
        for s in (1, 3, 5, 2, 4)
    ]
    view = build_view(base.model_copy(update={"counter_case": rebuttals}))
    assert [r.argument for r in view.counter_case] == ["arg 5", "arg 4", "arg 3"]


def test_counter_case_shows_target_claim_text() -> None:
    view = build_view(sample_report())
    first = view.counter_case[0]
    assert [t.claim_id for t in first.targets] == ["claim_thesis", "claim_fatigue"]
    assert first.targets[1].text == "That is simply not true"


# ---------------------------------------------------------------------------
# Quoting
# ---------------------------------------------------------------------------


def test_quotes_come_from_document_when_given() -> None:
    report = sample_report()
    polish = report.polish[0]
    assert polish.span is not None
    with_doc = build_view(report, OPINION_DOC)
    assert with_doc.polish[0].items[0].quote == "their is"
    assert with_doc.polish[0].label == "Paragraph 4"
    # Without the document, a finding with no linked claim has no quote.
    without = build_view(report)
    assert without.polish[0].items[0].quote is None
    assert without.polish[0].label == "Copy edits"
    # Findings linked to a claim with the same span fall back to the claim text.
    assert (
        without.must_fix[0].quote
        == "their press created more goals than any other team that season"
    )


def test_polish_groups_by_paragraph_in_document_order() -> None:
    doc = OPINION_DOC
    edits = [
        Finding(
            id=f"find_{i}",
            agent=AgentName.COPYEDIT,
            severity=Severity.STYLE,
            span=span_of(doc.text, quote),
            message=f"edit {i}",
        )
        for i, quote in enumerate(["their is", "Critics say", "High pressing is the single"])
    ]
    edits.append(
        Finding(id="find_doc", agent=AgentName.COPYEDIT, severity=Severity.STYLE, message="global")
    )
    report = sample_report().model_copy(update={"polish": edits})
    groups = build_view(report, doc).polish
    assert [g.label for g in groups] == [
        "Paragraph 2",
        "Paragraph 3",
        "Paragraph 4",
        "Whole document",
    ]


def test_long_quotes_are_truncated() -> None:
    text = "word " * 200
    doc = Document.from_text(text, id="doc_long")
    finding = Finding(
        agent=AgentName.FACTCHECK,
        severity=Severity.UNSUPPORTED,
        span=Span(start=0, end=len(text)),
        message="long",
    )
    report = Report(
        document_id=doc.id, profile=Profile.OPINION, verdict_line="v", must_fix=[finding]
    )
    quote = build_view(report, doc).must_fix[0].quote
    assert quote is not None
    assert len(quote) == MAX_QUOTE_CHARS
    assert quote.endswith("…")


def test_out_of_range_span_is_not_quoted() -> None:
    finding = Finding(
        agent=AgentName.FACTCHECK,
        severity=Severity.UNSUPPORTED,
        span=Span(start=0, end=10_000),
        message="m",
    )
    report = Report(
        document_id=SHORT_DOC.id, profile=Profile.OPINION, verdict_line="v", must_fix=[finding]
    )
    assert build_view(report, SHORT_DOC).must_fix[0].quote is None


def test_mismatched_document_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        render_markdown(sample_report(), SHORT_DOC)


# ---------------------------------------------------------------------------
# Safety: untrusted text and URLs
# ---------------------------------------------------------------------------


def test_html_escapes_untrusted_text() -> None:
    report, doc = _hostile_report()
    out = render_html(report, doc)
    assert "<script" not in out
    assert "<iframe" not in out
    assert "<img" not in out
    assert "<b>" not in out
    assert "&lt;script&gt;alert(&#39;msg&#39;)&lt;/script&gt;" in out
    assert "&lt;script&gt;alert(&#39;doc&#39;)&lt;/script&gt;" in out


def test_html_never_links_unsafe_urls() -> None:
    report, doc = _hostile_report()
    out = render_html(report, doc)
    hrefs = re.findall(r'href="([^"]*)"', out)
    assert hrefs, "expected the https source to be linked"
    assert all(h.startswith(("http://", "https://")) for h in hrefs)
    assert "javascript:" not in "".join(hrefs)
    assert "link removed: unsafe URL javascript:alert(document.cookie)" in out
    # The quote character in the URL cannot break out of the attribute.
    assert 'onmouseover="' not in out
    assert "https://example.org/a%22onmouseover=%22alert(1)" not in out  # parens encoded too
    assert "https://example.org/a%22onmouseover=%22alert%281%29" in out


def test_html_email_is_self_contained() -> None:
    out = render_html(sample_report(), OPINION_DOC)
    lowered = out.lower()
    for forbidden in ("<script", "<style", "<link", "<img", "src=", "@import", "url("):
        assert forbidden not in lowered


def test_markdown_neutralizes_injection() -> None:
    report, doc = _hostile_report()
    out = render_markdown(report, doc)
    # No live links other than the escaped-safe https source.
    links = re.findall(r"(?<!\\)\[[^\]]*(?<!\\)\]\(([^)]*)\)", out)
    assert links == ["https://example.org/a%22onmouseover=%22alert%281%29"]
    assert "javascript:alert(document.cookie)" not in re.findall(r"\]\(([^)]*)\)", out)
    # Raw HTML is escaped.
    assert re.search(r"(?<!\\)<script", out) is None
    assert re.search(r"(?<!\\)<iframe", out) is None
    # Newlines in a message cannot start a new block.
    assert not any(line.startswith("# Injected") for line in out.splitlines())
    assert "\\# not a heading" in out
    # Emphasis in suggestions is escaped.
    assert "Use \\*\\*bold\\*\\* \\_here\\_" in out


def test_text_flags_unsafe_urls_and_keeps_one_line_messages() -> None:
    report, doc = _hostile_report()
    out = render_text(report, doc)
    assert "link removed: unsafe URL javascript:alert(document.cookie)" in out
    assert not any(line.startswith("# Injected") for line in out.splitlines())


def test_missing_rebuttal_target_is_labelled() -> None:
    report, doc = _hostile_report()
    for render in ALL_RENDERERS:
        assert "claim_missing (not in ledger)" in render(report, doc).replace("\\_", "_")


def test_email_subject_is_single_line_and_capped() -> None:
    report, doc = _hostile_report()
    email = render_email(report, doc)
    assert "\r" not in email.subject and "\n" not in email.subject
    assert email.subject.startswith("Review Desk report: Not ready Bcc:")
    long = empty_report().model_copy(update={"verdict_line": "x" * 500})
    assert len(email_subject(long)) == 120
    assert email_subject(empty_report()) == "Review Desk report: Ready: no issues found."


def test_control_characters_are_stripped() -> None:
    assert clean("a\x00b\x1b[31mc‮d\te\nf") == "ab[31mcd\te\nf"
    assert one_line("  a \n\n b\t c ") == "a b c"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.org/x?a=1&b=2", "https://example.org/x?a=1&b=2"),
        ("http://example.org/a b", None),
        ("HTTPS://Example.org/", "HTTPS://Example.org/"),
        ("javascript:alert(1)", None),
        ("JaVaScRiPt:alert(1)", None),
        ("data:text/html,<script>", None),
        ("vbscript:msgbox", None),
        ("//example.org/protocol-relative", None),
        ("/relative/path", None),
        ("https:///no-host", None),
        ("", None),
        ("https://example.org/\x00", None),
        ("https://example.org/(x)<y>", "https://example.org/%28x%29%3Cy%3E"),
    ],
)
def test_safe_url(url: str, expected: str | None) -> None:
    assert safe_url(url) == expected


def test_md_escapes_leading_block_markers() -> None:
    assert md("# h") == "\\# h"
    assert md("1. item") == "\\1. item"
    assert md("- item") == "\\- item"
    assert md("> q") == "\\> q"
    assert md("plain text, 2019-20.") == "plain text, 2019-20."
