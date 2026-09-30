"""Sample documents, a populated ledger and a populated report for tests.

The documents are short, invented texts with deliberate defects (a wrong
number, an unsupported claim, a weak argument, a clumsy sentence) so every
agent has something to find. Spans in the sample ledger are computed with
``str.index`` so they always round-trip exactly.
"""

from __future__ import annotations

from datetime import UTC, datetime

from reviewdesk.contracts.models import (
    AgentName,
    Claim,
    ClaimLedger,
    ClaimType,
    Document,
    Evidence,
    Finding,
    Profile,
    Rebuttal,
    RebuttalTarget,
    Report,
    ResearchAction,
    ResearchStep,
    Severity,
    Span,
    Usage,
    Verdict,
)

FIXED_TIME = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
"""Deterministic timestamp for snapshot tests."""

OPINION_TEXT = """\
# Why high pressing wins titles

High pressing is the single most important tactic in modern football. \
Liverpool won the 2019-20 Premier League with 99 points, and their press \
created more goals than any other team that season. Teams that press high \
concede fewer shots, because opponents never get time on the ball.

Critics say pressing exhausts players by February. That is simply not true, \
and every club should copy the approach immediately.

The data is clear and their is no real debate left.
"""

DESIGN_DOC_TEXT = """\
# Design: Move sessions to Redis

## Context
Our Postgres sessions table handles 4,000 writes per second at peak and is \
the top source of lock contention.

## Proposal
Store sessions in a single Redis instance with a 30-day TTL. Redis handles \
over 100,000 writes per second on commodity hardware, so capacity is solved.

## Rollout
Switch all traffic on Monday.
"""

REPORT_TEXT = """\
# Q3 remote work survey

Of 1,200 employees surveyed, 68% said they prefer hybrid work. Remote staff \
filed 12% fewer support tickets than office staff. The company's founder, \
Ada Lovelace, said in 2021 that "offices are for meetings, not work."

Therefore remote work causes higher productivity.
"""

ESSAY_TEXT = """\
# The case for public libraries

Public libraries are the most efficient public service ever invented. The \
first free public library in the United States opened in Peterborough, New \
Hampshire, in 1833. Libraries today lend far more than books: tools, seeds \
and internet hotspots.

Cutting library budgets saves little and costs communities a lot.
"""

SHORT_TEXT = "Water boils at 90 degrees Celsius at sea level."

OPINION_DOC = Document.from_text(OPINION_TEXT, id="doc_opinion")
DESIGN_DOC = Document.from_text(DESIGN_DOC_TEXT, id="doc_design")
REPORT_DOC = Document.from_text(REPORT_TEXT, id="doc_report")
ESSAY_DOC = Document.from_text(ESSAY_TEXT, id="doc_essay")
SHORT_DOC = Document.from_text(SHORT_TEXT, id="doc_short")


def sample_documents() -> list[Document]:
    """Five sample documents covering opinion, design doc, report and essay."""
    return [OPINION_DOC, DESIGN_DOC, REPORT_DOC, ESSAY_DOC, SHORT_DOC]


def span_of(text: str, quote: str) -> Span:
    """Span of the first occurrence of ``quote`` in ``text``."""
    start = text.index(quote)
    return Span(start=start, end=start + len(quote))


def claim_for(doc: Document, quote: str, type: ClaimType, importance: float, *, id: str) -> Claim:
    """Build a claim whose span round-trips to ``quote`` in ``doc``."""
    return Claim(id=id, text=quote, span=span_of(doc.text, quote), type=type, importance=importance)


def _evidence(url: str, title: str, excerpt: str, *, primary: bool = False) -> Evidence:
    return Evidence(
        url=url, title=title, excerpt=excerpt, retrieved_at=FIXED_TIME, is_primary=primary
    )


EVIDENCE_PL_TABLE = _evidence(
    "https://www.premierleague.com/tables?season=2019-20",
    "Premier League Table 2019/20",
    "Liverpool: played 38, points 99.",
    primary=True,
)
EVIDENCE_PRESS_FATIGUE = _evidence(
    "https://example.org/sports-science/pressing-fatigue",
    "High-intensity pressing and late-season fatigue",
    "Teams in the top quartile for pressing intensity dropped 11% in pressing "
    "actions after February.",
)
EVIDENCE_PRESS_GOALS = _evidence(
    "https://example.org/stats/2019-20-pressing-goals",
    "Goals from high turnovers, 2019-20",
    "Manchester City scored the most goals from high turnovers in 2019-20.",
)


def sample_ledger() -> ClaimLedger:
    """A ledger for ``OPINION_DOC`` with every kind of entry populated.

    Contains a thesis, a supporting claim, three factual claims (verified,
    wrong, unsupported), rebuttals with and without evidence, a research
    trail, and a copy-edit finding overlapping a claim.
    """
    doc = OPINION_DOC
    ledger = ClaimLedger()
    thesis = claim_for(
        doc,
        "High pressing is the single most important tactic in modern football.",
        ClaimType.THESIS,
        1.0,
        id="claim_thesis",
    )
    supporting = claim_for(
        doc,
        "Teams that press high concede fewer shots, because opponents never get time on the ball.",
        ClaimType.SUPPORTING,
        0.7,
        id="claim_shots",
    )
    points = claim_for(
        doc,
        "Liverpool won the 2019-20 Premier League with 99 points",
        ClaimType.FACTUAL,
        0.6,
        id="claim_points",
    )
    goals = claim_for(
        doc,
        "their press created more goals than any other team that season",
        ClaimType.FACTUAL,
        0.8,
        id="claim_goals",
    )
    fatigue = claim_for(
        doc,
        "That is simply not true",
        ClaimType.FACTUAL,
        0.5,
        id="claim_fatigue",
    )
    debate = claim_for(
        doc,
        "The data is clear and their is no real debate left.",
        ClaimType.SUPPORTING,
        0.4,
        id="claim_debate",
    )
    for claim in (thesis, supporting, points, goals, fatigue, debate):
        ledger.add_claim(claim)

    ledger.set_verdict("claim_points", Verdict.VERIFIED, [EVIDENCE_PL_TABLE], confidence=0.95)
    ledger.set_verdict(
        "claim_goals",
        Verdict.WRONG,
        [EVIDENCE_PRESS_GOALS],
        confidence=0.8,
        note="Manchester City, not Liverpool, led this metric.",
    )
    ledger.set_verdict(
        "claim_fatigue",
        Verdict.UNSUPPORTED,
        [],
        confidence=0.4,
        note="Research cap reached without a settling source.",
    )
    ledger.add_research_step(
        "claim_goals",
        ResearchStep(
            agent=AgentName.FACTCHECK,
            iteration=1,
            action=ResearchAction.SEARCH,
            input="most goals from high press 2019-20 premier league",
            summary="Found a stats page ranking teams by goals from high turnovers.",
            evidence_urls=[EVIDENCE_PRESS_GOALS.url],
            at=FIXED_TIME,
        ),
    )
    ledger.add_rebuttal(
        Rebuttal(
            id="reb_fatigue",
            target_claim_ids=["claim_thesis", "claim_fatigue"],
            argument="Pressing volume measurably drops late in the season, so the "
            "tactic's value depends on squad depth.",
            evidence=[EVIDENCE_PRESS_FATIGUE],
            strength=4,
            target=RebuttalTarget.INTERPRETATION,
        )
    )
    ledger.add_rebuttal(
        Rebuttal(
            id="reb_copy",
            target_claim_ids=["claim_thesis"],
            argument="Copying a tactic without the personnel to execute it is a known "
            "failure mode.",
            evidence=[],
            strength=2,
        )
    )
    ledger.add_finding(
        Finding(
            id="find_their",
            agent=AgentName.COPYEDIT,
            severity=Severity.STYLE,
            span=span_of(doc.text, "their is"),
            message="'their' should be 'there'.",
            suggestion="there is",
        )
    )
    return ledger


def sample_report() -> Report:
    """A report with every section populated, built on ``sample_ledger``."""
    ledger = sample_ledger()
    doc = OPINION_DOC
    must_fix = [
        Finding(
            id="find_goals",
            agent=AgentName.FACTCHECK,
            severity=Severity.FACTUAL_ERROR,
            span=ledger.get("claim_goals").claim.span,
            message="Liverpool did not create the most goals from pressing in 2019-20.",
            evidence=[EVIDENCE_PRESS_GOALS],
            claim_ids=["claim_goals"],
        ),
        Finding(
            id="find_fatigue",
            agent=AgentName.FACTCHECK,
            severity=Severity.UNSUPPORTED,
            span=ledger.get("claim_fatigue").claim.span,
            message="No source supports dismissing pressing fatigue.",
            claim_ids=["claim_fatigue"],
        ),
    ]
    should_fix = [
        Finding(
            id="find_structure",
            agent=AgentName.STRUCTURE,
            severity=Severity.STRUCTURE,
            span=None,
            message="The piece asserts a conclusion before presenting any evidence.",
            suggestion="Move the data section before the recommendation.",
        )
    ]
    polish = [ledger.findings["find_their"]]
    originality = [
        Finding(
            id="find_orig",
            agent=AgentName.ORIGINALITY,
            severity=Severity.CONSIDER,
            span=span_of(doc.text, "High pressing is the single most important tactic"),
            message="Possible near-match found online.",
            evidence=[
                Evidence(
                    url="https://example.com/blog/pressing",
                    title="Pressing explained",
                    excerpt="High pressing is the single most important tactic in football.",
                    retrieved_at=FIXED_TIME,
                )
            ],
            heuristic=True,
        )
    ]
    counter_case = ledger.get("claim_thesis").rebuttals
    counts = {
        Severity.FACTUAL_ERROR.label: 1,
        Severity.UNSUPPORTED.label: 1,
        Severity.STRONG_REBUTTAL.label: 1,
        Severity.STRUCTURE.label: 1,
        Severity.STYLE.label: 1,
        Severity.CONSIDER.label: 2,
    }
    return Report(
        document_id=doc.id,
        profile=Profile.OPINION,
        verdict_line="Not ready: one factual error and one unsupported claim to fix.",
        counts=counts,
        must_fix=must_fix,
        counter_case=counter_case,
        should_fix=should_fix,
        polish=polish,
        originality=originality,
        ledger=ledger,
        usage=Usage(input_tokens=12_000, output_tokens=3_000, search_calls=9, llm_calls=14),
        notes=[],
        generated_at=FIXED_TIME,
    )


def empty_report() -> Report:
    """A report with every section empty."""
    return Report(
        document_id=SHORT_DOC.id,
        profile=Profile.OPINION,
        verdict_line="Ready: no issues found.",
        generated_at=FIXED_TIME,
    )
