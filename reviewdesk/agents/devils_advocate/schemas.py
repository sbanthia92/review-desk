"""Structured-output schemas for the devil's advocate's LLM calls.

These are private to the agent: they describe what the model returns, not
contract types. Numeric fields are deliberately unbounded here and clamped in
code, so a slightly out-of-range model reply degrades gracefully instead of
failing schema validation.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from reviewdesk.contracts import RebuttalTarget


class ResearchPlan(BaseModel):
    """Reply to ``devils_advocate.plan``: search queries for counter-evidence."""

    queries: list[str] = Field(default_factory=list)
    """Up to three queries, best first."""
    rationale: str = ""


class NextAction(StrEnum):
    """What the research loop should do after assessing sources."""

    STOP = "stop"
    REFINE = "refine"
    FOLLOW_LINK = "follow_link"


class FoundEvidence(BaseModel):
    """One counter-evidence excerpt the model found in a numbered source."""

    source_id: str
    """Source label from the prompt, e.g. ``"S1"``."""
    excerpt: str
    """Verbatim quote from the source. Checked against the source text."""
    is_primary: bool = False


class Assessment(BaseModel):
    """Reply to ``devils_advocate.assess``."""

    evidence: list[FoundEvidence] = Field(default_factory=list)
    next_action: NextAction = NextAction.STOP
    next_query: str | None = None
    follow_url: str | None = None
    summary: str = ""


class Angle(StrEnum):
    """Kind of objection. Design-doc reviews mostly use the last three."""

    COUNTER_EVIDENCE = "counter_evidence"
    ALTERNATIVE = "alternative"
    ASSUMPTION = "assumption"
    FAILURE_MODE = "failure_mode"


class RebuttalDraft(BaseModel):
    """One rebuttal as proposed by the model, before repair and validation."""

    target_claim_ids: list[str] = Field(default_factory=list)
    argument: str
    evidence_ids: list[str] = Field(default_factory=list)
    """Evidence labels from the prompt, e.g. ``["E1", "E3"]``."""
    strength: int = 3
    target: RebuttalTarget = RebuttalTarget.INTERPRETATION
    angle: Angle = Angle.COUNTER_EVIDENCE


class RebuttalSet(BaseModel):
    """Reply to ``devils_advocate.rebut``."""

    rebuttals: list[RebuttalDraft] = Field(default_factory=list)


class DebateReply(BaseModel):
    """Reply to ``devils_advocate.debate``."""

    argument: str
    concedes: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
