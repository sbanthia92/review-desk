"""Default plans, budget splitting and plan validation."""

from __future__ import annotations

import pytest

from reviewdesk.contracts import (
    PROFILE_AGENTS,
    AgentName,
    Budget,
    Profile,
)
from reviewdesk.orchestrator.planning import (
    default_plan,
    plan_totals,
    split_budget,
    validate_plan,
)
from reviewdesk.orchestrator.schemas import PlanClaimBudget, PlanReply
from reviewdesk.testing.fakes import OPINION_DOC, REPORT_DOC, sample_ledger
from tests.orchestrator.builders import bare_ledger

ALL = [a.value for a in AgentName]


def test_default_plan_uses_profile_agents_without_extractor() -> None:
    for profile in (Profile.OPINION, Profile.DESIGN_DOC):
        plan = default_plan(profile, bare_ledger(), Budget(), available=ALL)
        expected = [a for a in PROFILE_AGENTS[profile] if a is not AgentName.EXTRACTOR]
        assert plan.agents == expected
    design = default_plan(Profile.DESIGN_DOC, bare_ledger(), Budget(), available=ALL)
    assert AgentName.ORIGINALITY not in design.agents


def test_default_plan_only_schedules_available_agents() -> None:
    plan = default_plan(Profile.OPINION, bare_ledger(), Budget(), available=["factcheck"])
    assert plan.agents == [AgentName.FACTCHECK]


def test_default_plan_deep_research_and_budgets_within_job() -> None:
    ledger = bare_ledger()
    budget = Budget(max_search_calls=60, max_tokens=400_000)
    plan = default_plan(Profile.OPINION, ledger, budget, available=ALL, document=OPINION_DOC)
    assert plan.deep_research_claim_ids[0] == "claim_thesis"
    assert "claim_debate" not in plan.deep_research_claim_ids  # importance 0.4
    searches, tokens = plan_totals(plan, ledger)
    assert searches <= budget.max_search_calls
    assert tokens <= budget.max_tokens
    assert plan.budget_for("claim_thesis").max_iterations == 3
    assert plan.budget_for("claim_debate").max_iterations == 1


def test_number_heavy_text_gets_more_fact_check_depth() -> None:
    from reviewdesk.contracts import ClaimLedger, ClaimType
    from reviewdesk.testing.fakes import claim_for

    ledger = ClaimLedger()
    ledger.add_claim(
        claim_for(REPORT_DOC, "68% said they prefer hybrid work", ClaimType.FACTUAL, 0.3, id="c1")
    )
    light = default_plan(Profile.OPINION, ledger, Budget(), available=ALL)
    heavy = default_plan(Profile.OPINION, ledger, Budget(), available=ALL, document=REPORT_DOC)
    assert "c1" not in light.deep_research_claim_ids
    assert "c1" in heavy.deep_research_claim_ids


@pytest.mark.parametrize(
    ("searches", "tokens"),
    [(60, 400_000), (10, 50_000), (3, 5_000), (0, 0), (1, 100)],
)
def test_split_budget_never_exceeds_job(searches: int, tokens: int) -> None:
    claims = [e.claim for e in sample_ledger().entries.values()]
    budget = Budget(max_search_calls=searches, max_tokens=tokens)
    split = split_budget(claims, budget, ["claim_thesis", "claim_goals"])
    assert set(split) == {c.id for c in claims}
    assert sum(b.max_search_calls for b in split.values()) <= searches
    assert sum(b.max_tokens for b in split.values()) <= tokens
    assert all(1 <= b.max_iterations <= 3 for b in split.values())


def test_split_budget_favours_important_claims_when_short() -> None:
    claims = [e.claim for e in sample_ledger().entries.values()]
    split = split_budget(claims, Budget(max_search_calls=4, max_tokens=10_000), [])
    by_importance = sorted(claims, key=lambda c: -c.importance)
    calls = [split[c.id].max_search_calls for c in by_importance]
    assert calls == sorted(calls, reverse=True)
    assert calls[0] >= 1
    assert calls[-1] == 0  # least important claim is left unchecked


def _reply(**kwargs: object) -> PlanReply:
    data: dict[str, object] = {
        "agents": ["factcheck", "devils_advocate"],
        "deep_research_claim_ids": ["claim_goals"],
        "claim_budgets": [],
        "rationale": "ok",
    }
    data.update(kwargs)
    return PlanReply.model_validate(data)


def _cb(cid: str, iterations: int = 2, searches: int = 3, tokens: int = 1_000) -> PlanClaimBudget:
    return PlanClaimBudget(
        claim_id=cid, max_iterations=iterations, max_search_calls=searches, max_tokens=tokens
    )


VALIDATE_CASES = [
    (
        "valid plan accepted",
        _reply(),
        Profile.OPINION,
        ["factcheck", "devils_advocate", "originality"],
        "",
    ),
    (
        "unknown agent dropped",
        _reply(agents=["factcheck", "astrologer"]),
        Profile.OPINION,
        ["factcheck", "originality"],
        "",
    ),
    (
        "extractor dropped",
        _reply(agents=["extractor", "copyedit"]),
        Profile.OPINION,
        ["copyedit", "originality"],
        "",
    ),
    (
        "agent not allowed by profile dropped",
        _reply(agents=["originality", "structure"]),
        Profile.DESIGN_DOC,
        ["structure"],
        "",
    ),
    ("no agents left rejected", _reply(agents=["astrologer"]), Profile.OPINION, None, "no allowed"),
    (
        "iterations above cap rejected",
        _reply(claim_budgets=[_cb("claim_goals", iterations=5)]),
        Profile.OPINION,
        None,
        "iterations",
    ),
    (
        "zero iterations rejected",
        _reply(claim_budgets=[_cb("claim_goals", iterations=0)]),
        Profile.OPINION,
        None,
        "iterations",
    ),
    (
        "negative budget rejected",
        _reply(claim_budgets=[_cb("claim_goals", searches=-1)]),
        Profile.OPINION,
        None,
        "negative",
    ),
    (
        "search budgets over the job rejected",
        _reply(claim_budgets=[_cb("claim_goals", searches=50), _cb("claim_points", searches=20)]),
        Profile.OPINION,
        None,
        "search budget",
    ),
    (
        "token budgets over the job rejected",
        _reply(claim_budgets=[_cb("claim_goals", tokens=500_000)]),
        Profile.OPINION,
        None,
        "token budget",
    ),
]


@pytest.mark.parametrize(
    ("name", "reply", "profile", "agents", "reason"),
    VALIDATE_CASES,
    ids=[c[0] for c in VALIDATE_CASES],
)
def test_validate_plan(
    name: str, reply: PlanReply, profile: Profile, agents: list[str] | None, reason: str
) -> None:
    ledger = bare_ledger()
    plan, why = validate_plan(reply, profile=profile, ledger=ledger, budget=Budget(), available=ALL)
    if agents is None:
        assert plan is None
        assert reason in why
        return
    assert plan is not None
    assert [a.value for a in plan.agents] == agents
    assert plan.profile is profile


def test_validate_plan_drops_unknown_claims_and_keeps_budgets() -> None:
    reply = _reply(
        deep_research_claim_ids=["claim_goals", "claim_nope", "claim_goals"],
        claim_budgets=[_cb("claim_goals", iterations=3, searches=4), _cb("claim_nope")],
    )
    plan, _ = validate_plan(
        reply, profile=Profile.OPINION, ledger=bare_ledger(), budget=Budget(), available=ALL
    )
    assert plan is not None
    assert plan.deep_research_claim_ids == ["claim_goals"]
    assert set(plan.claim_budgets) == {"claim_goals"}
    assert plan.budget_for("claim_goals").max_search_calls == 4
    assert plan.budget_for("claim_points").max_iterations == 1  # unlisted claims get a single pass


def test_validate_plan_respects_registry() -> None:
    plan, _ = validate_plan(
        _reply(agents=["factcheck", "devils_advocate"]),
        profile=Profile.OPINION,
        ledger=bare_ledger(),
        budget=Budget(),
        available=["devils_advocate"],
    )
    assert plan is not None
    assert plan.agents == [AgentName.DEVILS_ADVOCATE]


def test_originality_is_required_for_opinion_only_when_available() -> None:
    ledger = bare_ledger()
    reply = _reply(agents=["factcheck"])
    plan, _ = validate_plan(
        reply, profile=Profile.OPINION, ledger=ledger, budget=Budget(), available=["factcheck"]
    )
    assert plan is not None
    assert [a.value for a in plan.agents] == ["factcheck"]  # not registered: not forced
    plan, _ = validate_plan(
        reply, profile=Profile.DESIGN_DOC, ledger=ledger, budget=Budget(), available=ALL
    )
    assert plan is not None
    assert "originality" not in [a.value for a in plan.agents]
