"""Structured-output schemas the fact-checker asks the LLM for.

These are internal to the fact-checker; they never leave the package. Code
re-validates everything the model returns (URLs must be ones we retrieved,
quotes must appear in the retrieved text, verdicts must be settled by the
evidence), so a confused or manipulated model cannot mark a claim verified on
its own say-so.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    """Reply to ``factcheck.queries``: search queries for one claim, best first."""

    queries: list[str] = Field(default_factory=list)
    rationale: str = ""


class Stance(StrEnum):
    """How one source bears on the claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    IRRELEVANT = "irrelevant"


class NextStep(StrEnum):
    """What the research loop should do after assessing sources."""

    STOP = "stop"
    """Nothing more to gain (or the claim is settled)."""
    REFINE = "refine"
    """Search again with ``next_query``."""
    FOLLOW = "follow"
    """Fetch ``follow_url`` (usually the primary source a page cites)."""


class SourceAssessment(BaseModel):
    """The model's reading of one retrieved source."""

    url: str
    stance: Stance
    is_primary: bool = False
    quote: str = ""
    """Verbatim sentence from the source that supports the stance."""


class Assessment(BaseModel):
    """Reply to ``factcheck.assess``: per-source stances and the next step."""

    sources: list[SourceAssessment] = Field(default_factory=list)
    next_step: NextStep = NextStep.STOP
    next_query: str | None = None
    follow_url: str | None = None
    summary: str = ""
    unverified_parts: list[str] = Field(default_factory=list)
    """Checkable parts of the claim that no source seen so far confirms or
    contradicts (a few words each). Empty when every part is covered."""


class PartVerdict(BaseModel):
    """The judge's verdict on one checkable part of a claim."""

    part: str
    verdict: Literal["verified", "wrong", "unsupported"]


class Judgment(BaseModel):
    """Reply to ``factcheck.judge``: the final verdict proposal for one claim."""

    verdict: Literal["verified", "wrong", "unsupported"]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    explanation: str = ""
    correction: str | None = None
    """For wrong claims: what the evidence says instead."""
    evidence_urls: list[str] = Field(default_factory=list)
    parts: list[PartVerdict] = Field(default_factory=list)
    """A verdict for each checkable part of the claim (number, date, name,
    place, cause, sequence). The claim is verified only if every part is."""
    contested: bool = True
    """For a "wrong" verdict: True if any evidence supports the specific
    detail the contradicting evidence disputes. Defaults to True (the cautious
    reading) when the model does not say."""


class DebateReply(BaseModel):
    """Reply to ``factcheck.debate``: one response to the opposing evidence."""

    argument: str
    concedes: bool = False
    evidence_urls: list[str] = Field(default_factory=list)
