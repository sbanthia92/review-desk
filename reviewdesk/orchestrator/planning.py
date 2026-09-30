"""Execution plans: the profile default, budget splitting and plan validation.

The planner LLM (tag ``orchestrator.plan``) proposes a plan; ``validate_plan``
checks it against the guardrails in code (allowed agents, the 3-iteration cap,
budgets summing within the job) and returns None when it is invalid, in which
case the caller uses ``default_plan``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

from reviewdesk.contracts import (
    MAX_RESEARCH_ITERATIONS,
    PROFILE_AGENTS,
    AgentName,
    Budget,
    Claim,
    ClaimBudget,
    ClaimLedger,
    ClaimType,
    Document,
    ExecutionPlan,
    Profile,
)
from reviewdesk.orchestrator.labels import agent_rank
from reviewdesk.orchestrator.schemas import PlanReply

CLAIM_SEARCH_SHARE = 0.8
"""Share of the job's search calls the default plan splits across claims."""
CLAIM_TOKEN_SHARE = 0.6
"""Share of the job's tokens the default plan splits across claims (rest: agents' overhead)."""
MAX_CLAIM_TOKENS = 30_000
"""Per-claim token cap in the default plan."""
SEARCHES_PER_ITERATION = 3
"""Per-claim search cap in the default plan is this times the claim's iterations."""
DEEP_IMPORTANCE = 0.5
"""Claims at or above this importance get deep research in the default plan."""
MAX_DEEP_CLAIMS = 8
MAX_DEEP_CLAIMS_NUMBER_HEAVY = 12
UNLISTED_CLAIM_BUDGET = ClaimBudget(max_iterations=1, max_search_calls=1, max_tokens=2_000)
"""Budget a validated LLM plan gives claims it did not list."""
_MIN_WEIGHT = 0.05


def plannable_agents(profile: Profile, available: Iterable[str]) -> list[AgentName]:
    """Agents ``profile`` allows (minus the extractor) that are in ``available``."""
    present = set(available)
    return [a for a in PROFILE_AGENTS[profile] if a is not AgentName.EXTRACTOR and a in present]


def is_number_heavy(doc: Document) -> bool:
    """True if the text is dense with figures (adds fact-check depth)."""
    numbers = len(re.findall(r"\d[\d,.]*%?", doc.text))
    return numbers >= max(4, doc.word_count // 40)


def default_deep_claims(ledger: ClaimLedger, *, number_heavy: bool = False) -> list[str]:
    """Claim IDs that get the full research loop by default (most important first)."""
    cap = MAX_DEEP_CLAIMS_NUMBER_HEAVY if number_heavy else MAX_DEEP_CLAIMS
    chosen: list[str] = []
    thesis = ledger.thesis()
    if thesis is not None:
        chosen.append(thesis.id)
    for claim in ledger.claims():
        if len(chosen) >= cap:
            break
        important = claim.importance >= DEEP_IMPORTANCE
        if number_heavy and claim.type is ClaimType.FACTUAL:
            important = important or claim.importance >= DEEP_IMPORTANCE / 2
        if important and claim.id not in chosen:
            chosen.append(claim.id)
    return chosen


def split_budget(
    claims: Sequence[Claim], budget: Budget, deep_ids: Iterable[str]
) -> dict[str, ClaimBudget]:
    """Split the job budget across claims by importance.

    Deep claims get up to 3 iterations, others 1. Search calls and tokens are
    shared in proportion to importance from a pool (``CLAIM_SEARCH_SHARE`` and
    ``CLAIM_TOKEN_SHARE`` of the job); most important claims are served first
    and each gets at least one search while the pool lasts, so when budget runs
    short the least important claims get zero searches (left unchecked). The
    totals never exceed the job budget.
    """
    deep = set(deep_ids)
    ordered = sorted(claims, key=lambda c: (-c.importance, c.span.start, c.id))
    weights = {c.id: max(c.importance, _MIN_WEIGHT) for c in ordered}
    total = sum(weights.values()) or 1.0
    search_pool = int(budget.max_search_calls * CLAIM_SEARCH_SHARE)
    token_pool = int(budget.max_tokens * CLAIM_TOKEN_SHARE)
    search_left, tokens_left = search_pool, token_pool
    out: dict[str, ClaimBudget] = {}
    for claim in ordered:
        iterations = MAX_RESEARCH_ITERATIONS if claim.id in deep else 1
        share = weights[claim.id] / total
        searches = min(
            max(1, math.floor(search_pool * share)),
            SEARCHES_PER_ITERATION * iterations,
            search_left,
        )
        tokens = min(math.floor(token_pool * share), MAX_CLAIM_TOKENS, tokens_left)
        search_left -= searches
        tokens_left -= tokens
        out[claim.id] = ClaimBudget(
            max_iterations=iterations, max_search_calls=searches, max_tokens=tokens
        )
    return out


def default_plan(
    profile: Profile,
    ledger: ClaimLedger,
    budget: Budget,
    *,
    available: Iterable[str],
    document: Document | None = None,
    rationale: str = "profile default",
) -> ExecutionPlan:
    """The profile's default plan: every allowed, available agent; budget split by importance."""
    heavy = document is not None and is_number_heavy(document)
    deep = default_deep_claims(ledger, number_heavy=heavy)
    claims = [e.claim for e in ledger.entries.values()]
    return ExecutionPlan(
        profile=profile,
        agents=plannable_agents(profile, available),
        deep_research_claim_ids=deep,
        claim_budgets=split_budget(claims, budget, deep),
        default_claim_budget=ClaimBudget(max_iterations=1, max_search_calls=0, max_tokens=0),
        rationale=rationale,
    )


def plan_totals(plan: ExecutionPlan, ledger: ClaimLedger) -> tuple[int, int]:
    """Total (search calls, tokens) the plan allots across every ledger claim."""
    searches = tokens = 0
    for claim_id in ledger.entries:
        claim_budget = plan.budget_for(claim_id)
        searches += claim_budget.max_search_calls
        tokens += claim_budget.max_tokens
    return searches, tokens


def validate_plan(
    reply: PlanReply,
    *,
    profile: Profile,
    ledger: ClaimLedger,
    budget: Budget,
    available: Iterable[str],
) -> tuple[ExecutionPlan | None, str]:
    """Check a proposed plan against the guardrails.

    Repairs what is safe to repair (drops unknown or disallowed agents and
    unknown claim IDs, deduplicates) and rejects the rest: no runnable agents,
    iterations outside 1-3, negative budgets, or per-claim budgets summing past
    the job budget. Returns ``(plan, "")`` or ``(None, reason)``.
    """
    allowed = plannable_agents(profile, available)
    agents: list[AgentName] = []
    for raw in reply.agents:
        name = next((a for a in allowed if a.value == raw.strip().lower()), None)
        if name is not None and name not in agents:
            agents.append(name)
    if not agents:
        return None, "plan schedules no allowed agents"
    agents.sort(key=agent_rank)

    deep: list[str] = []
    for cid in reply.deep_research_claim_ids:
        if cid in ledger.entries and cid not in deep:
            deep.append(cid)

    budgets: dict[str, ClaimBudget] = {}
    for item in reply.claim_budgets:
        if item.claim_id not in ledger.entries:
            continue
        if not 1 <= item.max_iterations <= MAX_RESEARCH_ITERATIONS:
            return None, f"claim budget has {item.max_iterations} iterations (cap is 3)"
        if item.max_search_calls < 0 or item.max_tokens < 0:
            return None, "claim budget is negative"
        budgets[item.claim_id] = ClaimBudget(
            max_iterations=item.max_iterations,
            max_search_calls=item.max_search_calls,
            max_tokens=item.max_tokens,
        )

    plan = ExecutionPlan(
        profile=profile,
        agents=agents,
        deep_research_claim_ids=deep,
        claim_budgets=budgets,
        default_claim_budget=UNLISTED_CLAIM_BUDGET,
        rationale=" ".join(reply.rationale.split())[:500],
    )
    searches, tokens = plan_totals(plan, ledger)
    if searches > budget.max_search_calls:
        return None, "claim search budgets exceed the job's search budget"
    if tokens > budget.max_tokens:
        return None, "claim token budgets exceed the job's token budget"
    return plan, ""
