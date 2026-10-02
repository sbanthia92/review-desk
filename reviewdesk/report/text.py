"""Render a ``Report`` to plain text (the email's text/plain fallback).

Plain text needs no escaping; untrusted strings are already stripped of
control characters and collapsed to one line by the view builder. URLs are
printed as text and never presented as links; unsafe ones are flagged.
"""

from __future__ import annotations

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


def _heading(lines: list[str], title: str) -> None:
    lines.extend([title, "=" * len(title), ""])


def _source_line(source: SourceView, indent: str) -> str:
    if source.href is not None:
        line = f"{indent}- {source.title} <{source.href}>"
    else:
        line = f"{indent}- {source.title} (link removed: unsafe URL {source.url})"
    if source.is_primary:
        line += " (primary source)"
    if source.excerpt:
        line += f': "{source.excerpt}"'
    return line


def _sources(lines: list[str], sources: list[SourceView], indent: str = "   ") -> None:
    if sources:
        lines.append(f"{indent}Sources:")
        lines.extend(_source_line(s, indent + "  ") for s in sources)


def _findings(lines: list[str], findings: list[FindingView], *, show_severity: bool) -> None:
    if not findings:
        lines.extend([EMPTY_SECTION, ""])
        return
    for i, f in enumerate(findings, 1):
        title = f.severity.capitalize() if show_severity else "Finding"
        lines.append(f"{i}. {title}")
        if f.quote:
            lines.append(f'   Quote: "{f.quote}"')
        lines.append(f"   Problem: {f.message}")
        if f.suggestion:
            lines.append(f"   Suggestion: {f.suggestion}")
        _sources(lines, f.sources)
        lines.append("")


def render_view_text(view: ReportView) -> str:
    """Render an already-built view to plain text."""
    lines: list[str] = []
    _heading(lines, "Review Desk report")

    lines.append(f"Verdict: {view.verdict_line}")
    if view.counts:
        lines.append(
            "Findings by severity: " + ", ".join(f"{c.label}: {c.count}" for c in view.counts)
        )
    else:
        lines.append("Findings by severity: none")
    lines.append("")

    _heading(lines, "Must fix")
    _findings(lines, view.must_fix, show_severity=True)

    _heading(lines, "Could not verify")
    if view.unverified:
        lines.extend([UNVERIFIED_NOTE, ""])
    _findings(lines, view.unverified, show_severity=False)

    _heading(lines, "Strongest counter-case")
    if not view.counter_case:
        lines.extend([EMPTY_SECTION, ""])
    for i, reb in enumerate(view.counter_case, 1):
        lines.append(f"{i}. Strength {reb.strength}/5 (attacks the {reb.target_kind})")
        lines.append(f"   {reb.argument}")
        lines.append("   Targets:")
        for t in reb.targets:
            text = f'"{t.text}"' if t.text is not None else f"claim {t.claim_id} (not in ledger)"
            lines.append(f"     - {text}")
        if reb.sources:
            _sources(lines, reb.sources)
        else:
            lines.append("   No sources cited: treat this as a consideration.")
        lines.append("")

    _heading(lines, "Should fix")
    _findings(lines, view.should_fix, show_severity=False)

    _heading(lines, "Polish")
    if not view.polish:
        lines.extend([EMPTY_SECTION, ""])
    for group in view.polish:
        lines.append(f"{group.label}:")
        for item in group.items:
            if item.quote:
                edit = f'"{item.quote}"'
                if item.suggestion:
                    edit += f' -> "{item.suggestion}"'
                lines.append(f"  - {edit}: {item.message}")
            elif item.suggestion:
                lines.append(f'  - {item.message} (suggestion: "{item.suggestion}")')
            else:
                lines.append(f"  - {item.message}")
        lines.append("")

    _heading(lines, "Originality notes (heuristic)")
    if view.originality:
        lines.extend([ORIGINALITY_DISCLAIMER, ""])
    _findings(lines, view.originality, show_severity=False)

    _heading(lines, "Appendix: claim ledger")
    if not view.ledger:
        lines.extend([EMPTY_SECTION, ""])
    for row in view.ledger:
        lines.append(
            f"- [{row.verdict}] {row.type}, importance {row.importance}, id {row.claim_id}"
        )
        lines.append(f'  "{row.text}"')
        if row.note:
            lines.append(f"  Note: {row.note}")
        _sources(lines, row.sources, indent="  ")
        if row.rebuttal_count:
            lines.append(f"  Rebuttals: {row.rebuttal_count}")
    if view.ledger:
        lines.append("")

    u = view.usage
    lines.extend(["-" * 40, f"Profile: {view.profile}. Generated {view.generated_at}."])
    usage = (
        f"Usage: {u.total_tokens} tokens ({u.input_tokens} in, {u.output_tokens} out), "
        f"{u.llm_calls} model calls, {u.search_calls} search calls, "
        f"{u.fetch_calls} page fetches"
    )
    if u.seconds is not None:
        usage += f", {u.seconds}s"
    lines.append(usage + ".")
    if view.notes:
        lines.append("Notices:")
        lines.extend(f"  - {note}" for note in view.notes)

    return "\n".join(lines).rstrip("\n") + "\n"


def render_text(report: Report, document: Document | None = None) -> str:
    """Render ``report`` to plain text (the email fallback part)."""
    return render_view_text(build_view(report, document))
