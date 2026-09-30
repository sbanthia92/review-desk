"""Structured-output schemas for the orchestrator's own LLM calls.

They are deliberately loose (plain ints and strings): the orchestrator
validates the content in code and falls back to a default when it is wrong.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClassifyReply(BaseModel):
    """Reply to ``orchestrator.classify``: which profile fits the document."""

    profile: Literal["opinion", "design_doc"]
    reason: str = ""


class PlanClaimBudget(BaseModel):
    """One claim's research budget as proposed by the planner."""

    claim_id: str
    max_iterations: int
    max_search_calls: int
    max_tokens: int


class PlanReply(BaseModel):
    """Reply to ``orchestrator.plan``: agents, deep-research claims, budgets."""

    agents: list[str]
    deep_research_claim_ids: list[str] = Field(default_factory=list)
    claim_budgets: list[PlanClaimBudget] = Field(default_factory=list)
    rationale: str = ""


class RulingReply(BaseModel):
    """Reply to ``orchestrator.debate_ruling``: who wins and the final verdict."""

    winner: Literal["factcheck", "devils_advocate"]
    final_verdict: Literal["verified", "wrong", "unsupported"]
    ruling: str
    reasoning: str
