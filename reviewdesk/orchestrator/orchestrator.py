"""The orchestrator: classify, extract, plan, review in parallel, react, resolve, report.

``Orchestrator.review`` runs one job end to end on the agents in an
``AgentRegistry``. It never fails because an agent failed: a raising or
erroring agent becomes a report note ("fact-check unavailable"), and hitting
the job's ``max_seconds`` returns partial results with a note.

All agents share one ``BudgetMeter`` (``ReviewContext.__post_init__`` builds a
fresh one, so the orchestrator reassigns ``ctx.meter``). Parallel agents read
one ledger snapshot; their results are applied afterwards in a fixed order
(``AGENT_ORDER``), so the outcome does not depend on completion order.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from reviewdesk.contracts import (
    AddRebuttal,
    AgentName,
    AgentResult,
    Budget,
    BudgetMeter,
    ClaimLedger,
    ClaimType,
    DebatePosition,
    Debater,
    Document,
    Evidence,
    ExecutionPlan,
    ModelTier,
    Profile,
    ProgressCallback,
    ProgressStep,
    Rechecker,
    Report,
    ReviewContext,
    SetVerdict,
    Usage,
    Verdict,
)
from reviewdesk.orchestrator import prompts
from reviewdesk.orchestrator.assemble import build_report
from reviewdesk.orchestrator.classify import classify
from reviewdesk.orchestrator.conflicts import Decision, rebuttal_finding_id, resolve_conflicts
from reviewdesk.orchestrator.labels import agent_rank, label
from reviewdesk.orchestrator.planning import default_plan, plannable_agents, validate_plan
from reviewdesk.orchestrator.progress import ProgressTracker
from reviewdesk.orchestrator.reactions import (
    apply_ruling,
    counter_evidence,
    debate_candidates,
    debate_record,
    fallback_ruling,
    recheck_candidates,
    ruling_is_consistent,
)
from reviewdesk.orchestrator.registry import AgentRegistry
from reviewdesk.orchestrator.schemas import PlanReply, RulingReply

log = logging.getLogger(__name__)

T = TypeVar("T")

PLAN_TOKEN_ESTIMATE = 3_000
RULING_TOKEN_ESTIMATE = 2_500
ERROR_NOTE_CHARS = 300


class _TimedOut:
    """Marker for work cut off by the job's time limit."""


TIMED_OUT = _TimedOut()

Outcome = Any
"""What a bounded call yields: its result, the exception it raised, or ``TIMED_OUT``."""


@dataclass
class _Job:
    """Mutable state of one review job."""

    document: Document
    budget: Budget
    focus: str | None
    style_guide: str | None
    tracker: ProgressTracker
    meter: BudgetMeter
    deadline: float
    ledger: ClaimLedger = field(default_factory=ClaimLedger)
    profile: Profile = Profile.OPINION
    plan: ExecutionPlan | None = None
    notes: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    ok: set[str] = field(default_factory=set)
    """Agents that returned a result (possibly with ``error``) this job."""
    unfinished: list[str] = field(default_factory=list)
    """Labels of work cut off by the time limit."""
    decisions: list[Decision] = field(default_factory=list)

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def spend(self, usage: Usage) -> None:
        self.usage = self.usage + usage
        self.meter.charge(usage)


def _max_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=max(a.input_tokens, b.input_tokens),
        output_tokens=max(a.output_tokens, b.output_tokens),
        search_calls=max(a.search_calls, b.search_calls),
        fetch_calls=max(a.fetch_calls, b.fetch_calls),
        llm_calls=max(a.llm_calls, b.llm_calls),
    )


def _error_note(name: str, error: str) -> str:
    text = " ".join(error.split())[:ERROR_NOTE_CHARS]
    lead = label(name)
    return text if text.lower().startswith(lead.lower()) else f"{lead}: {text}"


class Orchestrator:
    """Runs reviews on the agents of one ``AgentRegistry``.

    ``budget`` is the default job budget (``Budget()`` if omitted); ``review``
    may override it per job.
    """

    def __init__(self, registry: AgentRegistry, *, budget: Budget | None = None) -> None:
        self.registry = registry
        self.budget = budget or Budget()

    # ------------------------------------------------------------------ job

    async def review(
        self,
        document: Document,
        profile: Profile,
        on_progress: ProgressCallback | None = None,
        *,
        budget: Budget | None = None,
        focus: str | None = None,
        style_guide: str | None = None,
    ) -> Report:
        """Review ``document`` and return the report. Never raises for agent failures."""
        started = time.monotonic()
        job_budget = budget or self.budget
        job = _Job(
            document=document,
            budget=job_budget,
            focus=focus,
            style_guide=style_guide,
            tracker=ProgressTracker(on_progress),
            meter=BudgetMeter(job_budget),
            deadline=started + job_budget.max_seconds,
        )
        await self._classify(job, profile)
        await self._extract(job)
        plan = await self._plan(job)
        await self._review_parallel(job, plan)
        await self._react(job, plan)
        report = self._resolve_and_report(job)
        seconds = time.monotonic() - started
        report.usage = report.usage.model_copy(update={"seconds": seconds})
        job.tracker.emit(ProgressStep.DONE, "Review complete", 1.0)
        return report

    # -------------------------------------------------------------- helpers

    def _context(
        self, job: _Job, ledger: ClaimLedger, emit: ProgressCallback, plan: ExecutionPlan | None
    ) -> ReviewContext:
        ctx = ReviewContext(
            document=job.document,
            profile=job.profile,
            ledger=ledger,
            llm=self.registry.llm,
            search=self.registry.search,
            fetcher=self.registry.fetcher,
            budget=job.budget,
            emit_progress=emit,
            plan=plan,
            focus=job.focus,
            style_guide=job.style_guide,
        )
        ctx.meter = job.meter  # one shared meter per job
        return ctx

    async def _bounded(self, job: _Job, factory: Callable[[], Awaitable[T]]) -> T | _TimedOut:
        """Await ``factory()`` within the job's remaining time (``TIMED_OUT`` if cut off)."""
        remaining = job.remaining()
        if remaining <= 0:
            return TIMED_OUT
        try:
            async with asyncio.timeout(remaining) as scope:
                return await factory()
        except TimeoutError:
            if scope.expired():
                return TIMED_OUT
            raise

    async def _gather(
        self, job: _Job, factories: list[Callable[[], Awaitable[T]]]
    ) -> list[T | BaseException | _TimedOut]:
        """Run factories in parallel with ``asyncio.gather``; exceptions are returned."""
        results = await asyncio.gather(
            *(self._bounded(job, f) for f in factories), return_exceptions=True
        )
        return list(results)

    async def _llm(
        self,
        job: _Job,
        messages: list[Any],
        schema: type[Any],
        *,
        tier: ModelTier,
        tag: str,
        estimate: int,
        max_tokens: int,
    ) -> Any:
        """One orchestrator LLM call within budget and time; returns ``parsed`` or None."""
        try:
            job.meter.require_tokens(estimate)
        except Exception:
            return None

        async def call() -> Any:
            return await self.registry.llm.complete(
                messages, schema=schema, model_tier=tier, tag=tag, max_tokens=max_tokens
            )

        try:
            response = await self._bounded(job, call)
        except Exception as exc:
            log.info("orchestrator: %s failed (%s)", tag, type(exc).__name__)
            return None
        if isinstance(response, _TimedOut):
            return None
        job.spend(response.usage)
        parsed = response.parsed
        return parsed if isinstance(parsed, schema) else None

    def _absorb(self, job: _Job, name: str, outcome: Outcome) -> AgentResult | None:
        """Turn one agent outcome into ledger changes and notes. Returns the result if any."""
        if isinstance(outcome, _TimedOut):
            job.unfinished.append(label(name))
            return None
        if isinstance(outcome, BaseException) or not isinstance(outcome, AgentResult):
            kind = type(outcome).__name__
            log.warning("orchestrator: agent %s failed (%s)", name, kind)
            job.note(f"{label(name)} unavailable")
            return None
        job.ok.add(name)
        job.usage = job.usage + outcome.usage
        skipped = self._apply(job.ledger, outcome)
        if skipped:
            job.note(f"{label(name)}: {skipped} invalid ledger update(s) skipped")
        for note in outcome.notes:
            job.note(note)
        if outcome.error:
            job.note(_error_note(name, outcome.error))
        return outcome

    @staticmethod
    def _apply(ledger: ClaimLedger, result: AgentResult) -> int:
        """Apply a result update by update; invalid updates are skipped and counted."""
        skipped = 0
        dropped_findings: set[str] = set()
        for update in result.ledger_updates:
            try:
                ledger.apply(update)
            except (KeyError, ValueError):
                skipped += 1
                if isinstance(update, AddRebuttal):
                    dropped_findings.add(rebuttal_finding_id(update.rebuttal.id))
        for finding in result.findings:
            if finding.id not in dropped_findings:
                ledger.add_finding(finding)
        return skipped

    # ------------------------------------------------------------ 1. classify

    async def _classify(self, job: _Job, profile: Profile) -> None:
        step = ProgressStep.CLASSIFYING
        job.tracker.emit(step, "Classifying the document", 0.0)
        result = await classify(
            job.document, profile, self.registry.llm, time_limit=job.remaining()
        )
        job.profile = result.profile
        if result.usage.llm_calls or result.usage.total_tokens:
            job.spend(result.usage)
        how = {"user": "chosen by you", "llm": "detected", "heuristic": "guessed"}[result.method]
        job.tracker.emit(step, f"Profile: {result.profile.value} ({how})", 1.0)

    # ------------------------------------------------------------- 2. extract

    async def _extract(self, job: _Job) -> None:
        step = ProgressStep.EXTRACTING
        name = AgentName.EXTRACTOR.value
        job.tracker.start_parts(step, [name], "Extracting claims")
        agent = self.registry.get(name)
        if agent is None:
            job.note(f"{label(name)} unavailable")
            job.tracker.part(step, name, 1.0, "No claim extractor configured")
            return
        ctx = self._context(job, ClaimLedger(), job.tracker.callback_for(step, name), None)
        [outcome] = await self._gather(job, [lambda: agent.run(ctx)])
        self._absorb(job, name, outcome)
        claims = len(job.ledger.entries)
        job.tracker.part(step, name, 1.0, f"Found {claims} claim{'s' if claims != 1 else ''}")

    # ---------------------------------------------------------------- 3. plan

    async def _plan(self, job: _Job) -> ExecutionPlan:
        step = ProgressStep.PLANNING
        job.tracker.emit(step, "Planning the review", 0.0)
        profile = job.profile
        available = [name for name in self.registry.agents]
        for name in plannable_agents(profile, [a.value for a in AgentName]):
            if name not in self.registry:
                job.note(f"{label(name)} not configured")
        fallback = default_plan(
            profile, job.ledger, job.budget, available=available, document=job.document
        )
        allowed = plannable_agents(profile, available)
        plan = fallback
        if allowed:
            reply = await self._llm(
                job,
                prompts.plan_messages(
                    profile, job.ledger.claims(), [a.value for a in allowed], job.budget, job.focus
                ),
                PlanReply,
                tier=ModelTier.STRONG,
                tag=prompts.TAG_PLAN,
                estimate=PLAN_TOKEN_ESTIMATE,
                max_tokens=2_000,
            )
            if isinstance(reply, PlanReply):
                proposed, reason = validate_plan(
                    reply,
                    profile=profile,
                    ledger=job.ledger,
                    budget=job.budget,
                    available=available,
                )
                if proposed is not None:
                    plan = proposed
                else:
                    log.info("orchestrator: plan rejected (%s); using profile default", reason)
                    plan = fallback.model_copy(
                        update={"rationale": f"profile default (plan rejected: {reason})"}
                    )
        job.plan = plan
        agents = ", ".join(label(a) for a in plan.agents) or "none"
        job.tracker.emit(step, f"Plan: {agents}", 1.0)
        return plan

    # -------------------------------------------------------------- 4. review

    async def _review_parallel(self, job: _Job, plan: ExecutionPlan) -> None:
        step = ProgressStep.REVIEWING
        names = [a.value for a in plan.agents if a.value in self.registry]
        job.tracker.start_parts(step, names, "Running reviewers")
        if not names:
            return
        snapshot = job.ledger.model_copy(deep=True)
        factories: list[Callable[[], Awaitable[AgentResult]]] = []
        for name in names:
            agent = self.registry.agents[name]
            ctx = self._context(job, snapshot, job.tracker.callback_for(step, name), plan)
            factories.append(self._runner(job, agent.run, ctx, step, name))
        outcomes = await self._gather(job, factories)
        by_name = dict(zip(names, outcomes, strict=True))
        for name in sorted(names, key=agent_rank):
            self._absorb(job, name, by_name[name])

    def _runner(
        self,
        job: _Job,
        run: Callable[[ReviewContext], Awaitable[AgentResult]],
        ctx: ReviewContext,
        step: ProgressStep,
        name: str,
    ) -> Callable[[], Awaitable[AgentResult]]:
        async def go() -> AgentResult:
            try:
                return await run(ctx)
            finally:
                job.tracker.part(step, name, 1.0, f"Finished {label(name)}")

        return go

    # --------------------------------------------------------------- 5. react

    async def _react(self, job: _Job, plan: ExecutionPlan) -> None:
        step = ProgressStep.REACTING
        job.tracker.emit(step, "Checking where reviewers disagree", 0.0)
        fc_name = AgentName.FACTCHECK.value
        da_name = AgentName.DEVILS_ADVOCATE.value
        fact_checker = self.registry.get(fc_name) if fc_name in job.ok else None
        advocate = self.registry.get(da_name) if da_name in job.ok else None
        if advocate is None:
            job.tracker.emit(step, "No reactions needed", 1.0)
            return
        if job.remaining() <= 0:
            if recheck_candidates(job.ledger) or debate_candidates(job.ledger):
                job.unfinished.append("reaction round")
            return
        if isinstance(fact_checker, Rechecker):
            await self._rechecks(job, plan, fact_checker)
        job.tracker.emit(step, "Re-checks done", 0.5)
        debaters = [a for a in (fact_checker, advocate) if isinstance(a, Debater)]
        if debaters:
            await self._debates(job, plan, fact_checker, advocate)
        job.tracker.emit(step, "Reactions done", 1.0)

    async def _rechecks(self, job: _Job, plan: ExecutionPlan, checker: Rechecker) -> None:
        candidates = recheck_candidates(job.ledger)
        if not candidates:
            return
        job.tracker.emit(
            ProgressStep.REACTING, f"Re-checking {len(candidates)} disputed claim(s)", 0.1
        )
        snapshot = job.ledger.model_copy(deep=True)
        ctx = self._context(job, snapshot, lambda _e: None, plan)
        ids = list(candidates)

        def factory(cid: str, evidence: list[Evidence]) -> Callable[[], Awaitable[AgentResult]]:
            return lambda: checker.recheck(ctx, cid, evidence)

        outcomes = await self._gather(job, [factory(cid, candidates[cid]) for cid in ids])
        failed = 0
        for cid, outcome in zip(ids, outcomes, strict=True):
            job.ledger.entries[cid].rechecked = True  # one re-check per claim, even if it failed
            if isinstance(outcome, _TimedOut):
                job.unfinished.append("re-check")
                continue
            if not isinstance(outcome, AgentResult):
                failed += 1
                continue
            if any(isinstance(u, SetVerdict) and u.claim_id == cid for u in outcome.ledger_updates):
                self._drop_fact_findings(job.ledger, cid)
            job.usage = job.usage + outcome.usage
            self._apply(job.ledger, outcome)
            if outcome.error:
                failed += 1
        if failed:
            job.note(f"re-check failed for {failed} claim(s); original verdicts kept")

    @staticmethod
    def _drop_fact_findings(ledger: ClaimLedger, claim_id: str) -> None:
        """Remove the fact-checker's findings for a claim about to get a new verdict."""
        for fid, finding in list(ledger.findings.items()):
            if finding.agent == AgentName.FACTCHECK and finding.claim_ids == [claim_id]:
                ledger.remove_finding(fid)

    async def _debates(
        self,
        job: _Job,
        plan: ExecutionPlan,
        fact_checker: Any,
        advocate: Any,
    ) -> None:
        claim_ids = debate_candidates(job.ledger)
        if not claim_ids:
            return
        job.tracker.emit(ProgressStep.REACTING, f"Debating {len(claim_ids)} disputed claim(s)", 0.6)
        snapshot = job.ledger.model_copy(deep=True)
        ctx = self._context(job, snapshot, lambda _e: None, plan)

        async def respond(debater: Any, cid: str, evidence: list[Evidence]) -> DebatePosition:
            position = await debater.respond(ctx, cid, evidence)
            if not isinstance(position, DebatePosition):
                raise TypeError("respond did not return a DebatePosition")
            return position

        def factory(
            debater: Any, cid: str, evidence: list[Evidence]
        ) -> Callable[[], Awaitable[DebatePosition]]:
            return lambda: respond(debater, cid, evidence)

        calls: list[Callable[[], Awaitable[DebatePosition]]] = []
        slots: list[tuple[str, str]] = []
        for cid in claim_ids:
            entry = snapshot.entries[cid]
            if isinstance(fact_checker, Debater):
                calls.append(factory(fact_checker, cid, counter_evidence(entry)))
                slots.append((cid, "factcheck"))
            if isinstance(advocate, Debater):
                calls.append(factory(advocate, cid, list(entry.verdict_evidence)))
                slots.append((cid, "devils_advocate"))
        outcomes = await self._gather(job, calls)
        positions: dict[tuple[str, str], DebatePosition] = {}
        for slot, outcome in zip(slots, outcomes, strict=True):
            if isinstance(outcome, DebatePosition):
                positions[slot] = outcome
            elif isinstance(outcome, _TimedOut):
                job.unfinished.append("debate")

        async def rule(cid: str) -> RulingReply:
            fc = positions.get((cid, "factcheck"))
            da = positions.get((cid, "devils_advocate"))
            entry = snapshot.entries[cid]
            reply = await self._llm(
                job,
                prompts.ruling_messages(
                    entry.claim, list(entry.verdict_evidence), counter_evidence(entry), fc, da
                ),
                RulingReply,
                tier=ModelTier.STRONG,
                tag=prompts.TAG_RULING,
                estimate=RULING_TOKEN_ESTIMATE,
                max_tokens=600,
            )
            if isinstance(reply, RulingReply) and ruling_is_consistent(reply):
                return reply
            return fallback_ruling(fc, da)

        rulings = await asyncio.gather(*(rule(cid) for cid in claim_ids))
        for cid, reply in zip(claim_ids, rulings, strict=True):
            record = debate_record(
                reply, positions.get((cid, "factcheck")), positions.get((cid, "devils_advocate"))
            )
            apply_ruling(job.ledger, cid, record)

    # ------------------------------------------------------ 6-7. resolve, report

    def _resolve_and_report(self, job: _Job) -> Report:
        job.tracker.emit(ProgressStep.RESOLVING, "Resolving conflicts", 0.0)
        resolution = resolve_conflicts(job.ledger)
        job.decisions = resolution.decisions
        log.debug(
            "orchestrator: %d conflict decision(s): %s",
            len(resolution.decisions),
            [(d.rule, d.action, d.ids) for d in resolution.decisions],
        )
        job.tracker.emit(
            ProgressStep.RESOLVING, f"Resolved {len(resolution.decisions)} conflict(s)", 1.0
        )
        job.tracker.emit(ProgressStep.REPORTING, "Writing the report", 0.0)
        fc_name = AgentName.FACTCHECK.value
        if job.plan is not None and AgentName.FACTCHECK in job.plan.agents and fc_name in job.ok:
            unchecked = sum(
                1
                for e in resolution.ledger.entries.values()
                if e.claim.type is ClaimType.FACTUAL and e.verdict is Verdict.UNCHECKED
            )
            if unchecked:
                job.note(f"partial fact-check: {unchecked} claim(s) unchecked")
        if job.unfinished:
            unfinished = ", ".join(dict.fromkeys(job.unfinished))
            job.note(f"partial results: time limit reached ({unfinished} did not finish)")
        partial = bool(job.unfinished) or any(n.endswith("unavailable") for n in job.notes)
        usage = _max_usage(job.usage, job.meter.used)
        report = build_report(
            job.document,
            job.profile,
            resolution.ledger,
            usage=usage,
            notes=job.notes,
            partial=partial,
        )
        job.tracker.emit(ProgressStep.REPORTING, "Report ready", 1.0)
        return report
