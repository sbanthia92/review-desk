"""Offline demo pipeline for ``REVIEWDESK_FAKE=1``.

Runs the real orchestrator (``reviewdesk.pipeline.run_review``) on
``FakeAgent``s, ``FakeLLM``, ``FakeSearch`` and ``FakeFetcher``, so the MCP
server can be tried from Claude Desktop or Claude Code without any API key.
The findings are placeholders derived from the document's sentences; every
one says it came from demo mode. No network calls are made.
"""

from __future__ import annotations

import re

from reviewdesk.contracts import (
    AddClaim,
    AgentName,
    AgentResult,
    Claim,
    ClaimType,
    Finding,
    LedgerUpdate,
    ReviewContext,
    Severity,
    Span,
)
from reviewdesk.mcp_local.wiring import RegistryPipeline
from reviewdesk.orchestrator import AgentRegistry
from reviewdesk.testing.fakes import FakeAgent, FakeFetcher, FakeLLM, FakeSearch

DEMO_NOTE = "Demo mode (REVIEWDESK_FAKE=1): placeholder finding, no model was called."
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
MAX_DEMO_CLAIMS = 3


def _sentences(text: str) -> list[Span]:
    """Spans of non-blank sentences, trimmed of surrounding whitespace."""
    spans: list[Span] = []
    for match in _SENTENCE.finditer(text):
        chunk = match.group(0)
        stripped = chunk.strip()
        if len(stripped.split()) < 3 or stripped.startswith("#"):
            continue
        start = match.start() + (len(chunk) - len(chunk.lstrip()))
        spans.append(Span(start=start, end=start + len(stripped)))
    return spans


def _extract(ctx: ReviewContext) -> AgentResult:
    text = ctx.document.text
    updates: list[LedgerUpdate] = []
    for i, span in enumerate(_sentences(text)[:MAX_DEMO_CLAIMS]):
        kind = ClaimType.THESIS if i == 0 else ClaimType.SUPPORTING
        claim = Claim(
            id=f"claim_demo_{i + 1}",
            text=span.text_of(text),
            span=span,
            type=kind,
            importance=1.0 if i == 0 else 0.5,
        )
        updates.append(AddClaim(claim=claim))
    return AgentResult(agent=AgentName.EXTRACTOR, ledger_updates=updates)


def _copyedit(ctx: ReviewContext) -> AgentResult:
    spans = _sentences(ctx.document.text)
    findings = []
    if spans:
        findings.append(
            Finding(
                id="find_demo_copyedit",
                agent=AgentName.COPYEDIT,
                severity=Severity.STYLE,
                span=spans[-1],
                message=f"Consider tightening this sentence. {DEMO_NOTE}",
                suggestion="(demo) shorter wording",
            )
        )
    return AgentResult(agent=AgentName.COPYEDIT, findings=findings)


def _structure(ctx: ReviewContext) -> AgentResult:
    return AgentResult(
        agent=AgentName.STRUCTURE,
        findings=[
            Finding(
                id="find_demo_structure",
                agent=AgentName.STRUCTURE,
                severity=Severity.STRUCTURE,
                message=f"Consider a short summary up front. {DEMO_NOTE}",
            )
        ],
    )


def demo_registry(*, delay: float = 0.3) -> AgentRegistry:
    """A registry of fake agents; ``delay`` makes progress visible in clients."""
    agents = [
        FakeAgent(AgentName.EXTRACTOR, on_run=_extract, delay=delay, progress=["demo extract"]),
        FakeAgent(AgentName.FACTCHECK, delay=delay, progress=["demo fact-check (no search)"]),
        FakeAgent(AgentName.DEVILS_ADVOCATE, delay=delay, progress=["demo counter-case"]),
        FakeAgent(AgentName.COPYEDIT, on_run=_copyedit, delay=delay),
        FakeAgent(AgentName.STRUCTURE, on_run=_structure, delay=delay),
        FakeAgent(AgentName.ORIGINALITY, delay=delay),
    ]
    # An unscripted FakeLLM makes the orchestrator fall back to its heuristics.
    return AgentRegistry.of(agents, llm=FakeLLM(), search=FakeSearch(), fetcher=FakeFetcher())


def demo_pipeline(*, delay: float = 0.3) -> RegistryPipeline:
    """The demo ``ReviewRunner``."""
    return RegistryPipeline(lambda: demo_registry(delay=delay))
