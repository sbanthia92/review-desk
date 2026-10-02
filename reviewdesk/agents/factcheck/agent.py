"""The fact-checker agent: ``Agent``, ``Rechecker`` and ``Debater``.

For each factual claim in the ledger (most important first, at most
``max_claims``), it runs a bounded research loop (see ``research.py``) and
returns ``SetVerdict`` and ``AddResearchStep`` ledger updates plus findings
for wrong (``Severity.FACTUAL_ERROR``) and unsupported
(``Severity.UNSUPPORTED``) claims. It never mutates the ledger.

Deep research (the full 3-iteration loop) goes to the plan's
``deep_research_claim_ids``, or to the top ``default_deep_claims`` claims
when there is no plan. Other claims get a single search. Claims beyond the
cap, or that find the budget exhausted, are marked ``unchecked``. Provider
failures degrade the affected claims to ``unchecked`` and set
``AgentResult.error``; the job never fails because of the fact-checker.
"""

from __future__ import annotations

import asyncio
import contextlib

from reviewdesk.agents.factcheck import prompts
from reviewdesk.agents.factcheck.research import (
    ClaimOutcome,
    ClaimResearch,
    estimate_tokens,
)
from reviewdesk.agents.factcheck.schemas import DebateReply
from reviewdesk.agents.factcheck.sources import Source
from reviewdesk.contracts import (
    AgentName,
    AgentResult,
    Claim,
    ClaimBudget,
    ClaimType,
    DebatePosition,
    Evidence,
    LedgerUpdate,
    ModelTier,
    ProgressEvent,
    ProgressStep,
    ReviewContext,
    ReviewDeskError,
    SetVerdict,
    Usage,
    Verdict,
)

DEFAULT_MAX_CLAIMS = 20
DEFAULT_DEEP_CLAIMS = 5
DEFAULT_CONCURRENCY = 3
_OUT_DEBATE = 600


def _describe(exc: BaseException) -> str:
    """Short, safe error text (provider messages never contain keys or document text)."""
    message = " ".join(str(exc).split())[:200]
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class FactCheckAgent:
    """Verifies factual claims against retrieved sources."""

    def __init__(
        self,
        *,
        max_claims: int = DEFAULT_MAX_CLAIMS,
        default_deep_claims: int = DEFAULT_DEEP_CLAIMS,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self.max_claims = max_claims
        self.default_deep_claims = default_deep_claims
        self.concurrency = max(1, concurrency)

    @property
    def name(self) -> str:
        """``AgentName.FACTCHECK``."""
        return AgentName.FACTCHECK.value

    # -- Agent -------------------------------------------------------------

    async def run(self, ctx: ReviewContext) -> AgentResult:
        """Fact-check the ledger's factual claims. Never raises."""
        try:
            return await self._run(ctx)
        except Exception as exc:  # last-resort degradation; the job must go on
            return AgentResult(agent=self.name, error=f"fact-check unavailable ({_describe(exc)})")

    async def _run(self, ctx: ReviewContext) -> AgentResult:
        claims = ctx.ledger.claims(ClaimType.FACTUAL)
        if not claims:
            return AgentResult(agent=self.name)
        plan = ctx.plan
        if plan is not None:
            deep_ids = set(plan.deep_research_claim_ids)
        else:
            deep_ids = {c.id for c in claims[: min(self.default_deep_claims, self.max_claims)]}
        selected: list[Claim] = []
        skipped: list[Claim] = []
        for index, claim in enumerate(claims):
            (selected if index < self.max_claims or claim.id in deep_ids else skipped).append(claim)

        semaphore = asyncio.Semaphore(self.concurrency)
        started = 0
        researches: list[ClaimResearch] = []

        async def check(claim: Claim) -> ClaimOutcome:
            nonlocal started
            async with semaphore:
                started += 1
                ctx.emit_progress(
                    ProgressEvent(
                        step=ProgressStep.REVIEWING,
                        message=f"Fact-checking claim {started} of {len(selected)}",
                        percent=100.0 * (started - 1) / len(selected),
                    )
                )
                budget = plan.budget_for(claim.id) if plan is not None else ClaimBudget()
                research = ClaimResearch(ctx, claim, budget, deep=claim.id in deep_ids)
                researches.append(research)
                try:
                    return await research.run()
                except ReviewDeskError as exc:
                    return research.fail(exc)

        outcomes = await asyncio.gather(*(check(c) for c in selected))

        # Second look: claims in one document share a subject, so pages fetched
        # for one claim often settle another. No new searches or fetches.
        pool: dict[str, Source] = {}
        for research in researches:
            for source in research.sources.values():
                if source.kind == "page" and source.text and not source.suspicious:
                    pool.setdefault(source.url, source)

        async def revisit(research: ClaimResearch) -> None:
            async with semaphore:
                # A failed second look keeps the first verdict.
                with contextlib.suppress(ReviewDeskError):
                    await research.second_look(list(pool.values()))
                research.outcome.usage = research.usage

        unsettled = [r for r in researches if r.outcome.verdict is Verdict.UNSUPPORTED]
        if pool and unsettled:
            ctx.emit_progress(
                ProgressEvent(
                    step=ProgressStep.REVIEWING,
                    message=f"Re-reading gathered sources for {len(unsettled)} unsettled claim(s)",
                    percent=95.0,
                )
            )
            await asyncio.gather(*(revisit(r) for r in unsettled))

        updates: list[LedgerUpdate] = []
        findings = []
        usage = Usage()
        errors: list[str] = []
        for outcome in outcomes:
            updates.extend(outcome.updates)
            findings.extend(outcome.findings)
            usage = usage + outcome.usage
            if outcome.error:
                errors.append(outcome.error)
        for claim in skipped:
            updates.append(
                SetVerdict(
                    claim_id=claim.id,
                    verdict=Verdict.UNCHECKED,
                    note=f"Not checked: only the top {self.max_claims} factual claims are checked.",
                )
            )
        error = None
        if errors:
            error = (
                f"fact-check degraded: {len(errors)} of {len(selected)} claim(s) could not be "
                f"fully checked ({errors[0]})"
            )
        return AgentResult(
            agent=self.name, findings=findings, ledger_updates=updates, usage=usage, error=error
        )

    # -- Rechecker ---------------------------------------------------------

    async def recheck(
        self, ctx: ReviewContext, claim_id: str, counter_evidence: list[Evidence]
    ) -> AgentResult:
        """Re-judge one claim against counter-evidence. Never raises."""
        try:
            entry = ctx.ledger.get(claim_id)
        except KeyError:
            return AgentResult(agent=self.name, error=f"re-check: unknown claim {claim_id!r}")
        budget = ctx.plan.budget_for(claim_id) if ctx.plan is not None else ClaimBudget()
        research = ClaimResearch(ctx, entry.claim, budget, deep=True)
        try:
            outcome = await research.recheck(entry.verdict_evidence, counter_evidence)
        except ReviewDeskError as exc:
            # Keep the original verdict: a failed re-check changes nothing.
            return AgentResult(
                agent=self.name,
                ledger_updates=list(research.outcome.updates),
                usage=research.usage,
                error=f"re-check failed ({_describe(exc)})",
            )
        except Exception as exc:
            return AgentResult(agent=self.name, error=f"re-check failed ({_describe(exc)})")
        return AgentResult(
            agent=self.name,
            findings=outcome.findings,
            ledger_updates=outcome.updates,
            usage=outcome.usage,
            error=outcome.error,
        )

    # -- Debater -----------------------------------------------------------

    async def respond(
        self, ctx: ReviewContext, claim_id: str, opposing_evidence: list[Evidence]
    ) -> DebatePosition:
        """Answer the devil's advocate's evidence once. Never raises."""
        try:
            entry = ctx.ledger.get(claim_id)
        except KeyError:
            return DebatePosition(
                agent=self.name, claim_id=claim_id, argument="Unknown claim; no position."
            )
        ours = list(entry.verdict_evidence)
        fallback = DebatePosition(
            agent=self.name,
            claim_id=claim_id,
            argument=f"The fact-checker could not respond; its verdict ({entry.verdict}) "
            "stands on the evidence it cited.",
            evidence=ours,
        )
        messages = prompts.debate_messages(
            entry.claim, entry.verdict.value, entry.verdict_note, ours, list(opposing_evidence)
        )
        try:
            ctx.meter.require_tokens(estimate_tokens(messages, _OUT_DEBATE))
            response = await ctx.llm.complete(
                messages,
                schema=DebateReply,
                model_tier=ModelTier.MID,
                tag=prompts.TAG_DEBATE,
                max_tokens=_OUT_DEBATE,
            )
        except ReviewDeskError:
            return fallback
        ctx.meter.charge(response.usage)
        reply = response.parsed
        if not isinstance(reply, DebateReply):
            return fallback
        cited = [e for e in ours if e.url in set(reply.evidence_urls)] or ours
        argument = " ".join(reply.argument.split())[:1000] or fallback.argument
        return DebatePosition(
            agent=self.name,
            claim_id=claim_id,
            argument=argument,
            evidence=cited,
            concedes=reply.concedes,
        )
