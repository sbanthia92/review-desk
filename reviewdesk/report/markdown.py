"""Render a ``Report`` to markdown (the inline MCP tool result).

Untrusted text is backslash-escaped so it cannot inject links, images, raw
HTML, emphasis or block structure. Only ``http``/``https`` evidence URLs are
rendered as links.
"""

from __future__ import annotations

import re

from reviewdesk.contracts.models import Document, Report
from reviewdesk.report.view import (
    EMPTY_SECTION,
    ORIGINALITY_DISCLAIMER,
    UNVERIFIED_NOTE,
    FindingView,
    ReportView,
    SourceView,
    build_view,
)

_INLINE_SPECIAL = re.compile(r"([\\`*_\[\]<>|~&])")
_LEADING_BLOCK = re.compile(r"^(\s*)([#>+=-]|\d+[.)])")


def md(text: str) -> str:
    """Escape untrusted single-line text for inline markdown."""
    escaped = _INLINE_SPECIAL.sub(r"\\\1", text)
    return _LEADING_BLOCK.sub(lambda m: m.group(1) + "\\" + m.group(2), escaped)


def _source_line(source: SourceView) -> str:
    if source.href is not None:
        head = f"[{md(source.title)}]({source.href})"
    else:
        head = f"{md(source.title)} (link removed: unsafe URL {md(source.url)})"
    if source.is_primary:
        head += " (primary source)"
    if source.excerpt:
        head += f": “{md(source.excerpt)}”"
    return f"- {head}"


def _sources(lines: list[str], sources: list[SourceView]) -> None:
    if not sources:
        return
    lines.append("**Sources:**")
    lines.append("")
    lines.extend(_source_line(s) for s in sources)
    lines.append("")


def _finding(lines: list[str], number: int, finding: FindingView, *, show_severity: bool) -> None:
    title = finding.severity.capitalize() if show_severity else "Finding"
    lines.append(f"### {number}. {title}")
    lines.append("")
    if finding.quote:
        lines.append(f"> {md(finding.quote)}")
        lines.append("")
    lines.append(f"**Problem:** {md(finding.message)}")
    lines.append("")
    if finding.suggestion:
        lines.append(f"**Suggestion:** {md(finding.suggestion)}")
        lines.append("")
    _sources(lines, finding.sources)


def _findings(lines: list[str], findings: list[FindingView], *, show_severity: bool) -> None:
    if not findings:
        lines.extend([EMPTY_SECTION, ""])
        return
    for i, finding in enumerate(findings, 1):
        _finding(lines, i, finding, show_severity=show_severity)


def render_view_markdown(view: ReportView) -> str:
    """Render an already-built view to markdown."""
    lines: list[str] = ["# Review Desk report", ""]

    # 1. Verdict line and counts.
    lines.extend([f"**Verdict:** {md(view.verdict_line)}", ""])
    if view.counts:
        counts = " · ".join(f"{md(c.label)}: {c.count}" for c in view.counts)
        lines.append(f"**Findings by severity:** {counts}")
    else:
        lines.append("**Findings by severity:** none")
    lines.append("")

    # 2. Must fix.
    lines.extend(["## Must fix", ""])
    _findings(lines, view.must_fix, show_severity=True)

    # Could not verify: lower severity than must-fix.
    lines.extend(["## Could not verify", ""])
    if view.unverified:
        lines.extend([UNVERIFIED_NOTE, ""])
    _findings(lines, view.unverified, show_severity=False)

    # 3. Strongest counter-case.
    lines.extend(["## Strongest counter-case", ""])
    if not view.counter_case:
        lines.extend([EMPTY_SECTION, ""])
    for i, reb in enumerate(view.counter_case, 1):
        lines.append(f"### {i}. Strength {reb.strength}/5 (attacks the {reb.target_kind})")
        lines.append("")
        lines.extend([md(reb.argument), ""])
        lines.extend(["**Targets:**", ""])
        for target in reb.targets:
            if target.text is not None:
                lines.append(f"- “{md(target.text)}”")
            else:
                lines.append(f"- claim {md(target.claim_id)} (not in ledger)")
        lines.append("")
        if reb.sources:
            _sources(lines, reb.sources)
        else:
            lines.extend(["*No sources cited: treat this as a consideration.*", ""])

    # 4. Should fix.
    lines.extend(["## Should fix", ""])
    _findings(lines, view.should_fix, show_severity=False)

    # 5. Polish.
    lines.extend(["## Polish", ""])
    if not view.polish:
        lines.extend([EMPTY_SECTION, ""])
    for group in view.polish:
        lines.extend([f"### {md(group.label)}", ""])
        for item in group.items:
            message = md(item.message)
            if item.quote:
                edit = f"“{md(item.quote)}”"
                if item.suggestion:
                    edit += f" → “{md(item.suggestion)}”"
                lines.append(f"- {edit}: {message}")
            elif item.suggestion:
                lines.append(f"- {message} (suggestion: “{md(item.suggestion)}”)")
            else:
                lines.append(f"- {message}")
        lines.append("")

    # 6. Originality notes.
    lines.extend(["## Originality notes (heuristic)", ""])
    if view.originality:
        lines.extend([f"*{md(ORIGINALITY_DISCLAIMER)}*", ""])
    _findings(lines, view.originality, show_severity=False)

    # 7. Appendix: claim ledger.
    lines.extend(["## Appendix: claim ledger", ""])
    if not view.ledger:
        lines.extend([EMPTY_SECTION, ""])
    for row in view.ledger:
        lines.append(
            f"- **{row.verdict}** · {row.type} · importance {row.importance} · "
            f"id {md(row.claim_id)}"
        )
        lines.append(f"  > {md(row.text)}")
        if row.note:
            lines.append(f"  - Note: {md(row.note)}")
        for source in row.sources:
            lines.append("  " + _source_line(source))
        if row.rebuttal_count:
            lines.append(f"  - Rebuttals: {row.rebuttal_count}")
    if view.ledger:
        lines.append("")

    # Footer: usage and notices.
    u = view.usage
    lines.extend(["---", ""])
    usage = (
        f"Usage: {u.total_tokens} tokens ({u.input_tokens} in, {u.output_tokens} out) · "
        f"{u.llm_calls} model calls · {u.search_calls} search calls · "
        f"{u.fetch_calls} page fetches"
    )
    if u.seconds is not None:
        usage += f" · {u.seconds}s"
    lines.append(f"*Profile: {md(view.profile)} · Generated {view.generated_at}*  ")
    lines.append(f"*{usage}*")
    lines.append("")
    if view.notes:
        lines.extend(["**Notices:**", ""])
        lines.extend(f"- {md(note)}" for note in view.notes)
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def render_markdown(report: Report, document: Document | None = None) -> str:
    """Render ``report`` to markdown.

    Pass the reviewed ``document`` to quote spans from its text; otherwise
    claim text from the ledger is quoted where available.
    """
    return render_view_markdown(build_view(report, document))
