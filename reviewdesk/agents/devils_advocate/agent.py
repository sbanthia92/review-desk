"""The devil's advocate agent (T5).

Builds the strongest opposing case against the thesis and supporting claims
from retrieved counter-evidence.

Flow of ``run``:

1. Pick targets: thesis and supporting claims from the ledger, most important
   first (plan deep-research claims first when a plan exists), at most
   ``max_claims``.
2. For each target run a bounded research loop (hard cap 3 iterations):
   plan queries → search → fetch the best fresh hit → find-in-page → assess
   (quote counter-evidence, then stop / refine / follow a link). Every step is
   returned as an ``AddResearchStep``. Quoted excerpts are checked against
   the retrieved text, so the model cannot invent citations.
3. One ``rebut`` call turns the gathered evidence into ``Rebuttal``s. Target
   claim IDs are repaired against the ledger (case, brackets, claim text, or
   the claim the cited evidence was gathered for); rebuttals with no valid
   target are dropped, so every emitted rebuttal references at least one
   real claim. Strength is clamped to 1–5.
4. Each rebuttal becomes an ``AddRebuttal`` update plus a ``Finding``:
   ``Severity.STRONG_REBUTTAL`` when it cites evidence, ``Severity.CONSIDER``
   when it does not (the design doc's "no retrieved source → consider" rule,
   applied here explicitly and again by the orchestrator). Evidence-free
   rebuttals are kept, with empty ``evidence``, so the orchestrator can
   downgrade them.

Failures never raise: ``ProviderError`` and ``BudgetExceeded`` end the run
with ``AgentResult.error`` set and whatever partial output exists. A search
outage or an exhausted search budget stops research but still lets the agent
write (evidence-free) rebuttals if tokens remain.
"""

from __future__ import annotations

import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel

from reviewdesk.agents.devils_advocate import prompts
from reviewdesk.agents.devils_advocate.prompts import EvidenceView, SourceView
from reviewdesk.agents.devils_advocate.schemas import (
    Angle,
    Assessment,
    DebateReply,
    NextAction,
    RebuttalDraft,
    RebuttalSet,
    ResearchPlan,
)
from reviewdesk.agents.devils_advocate.text import (
    domain,
    find_in_page,
    keywords,
    locate_excerpt,
    normalize,
    urls_in,
)
from reviewdesk.contracts import (
    MAX_RESEARCH_ITERATIONS,
    AddRebuttal,
    AddResearchStep,
    AgentName,
    AgentResult,
    BudgetExceeded,
    Claim,
    ClaimBudget,
    ClaimType,
    DebatePosition,
    Evidence,
    FetchedPage,
    FetchError,
    Finding,
    LedgerUpdate,
    Message,
    ModelTier,
    ProgressEvent,
    ProviderError,
    Rebuttal,
    ResearchAction,
    ResearchStep,
    ReviewContext,
    ReviewDeskError,
    SchemaError,
    SearchResult,
    Severity,
    Usage,
    new_id,
    utcnow,
)

T = TypeVar("T", bound=BaseModel)

NAME = AgentName.DEVILS_ADVOCATE.value

_ANGLE_PREFIX = {
    Angle.COUNTER_EVIDENCE: "Counter-argument: ",
    Angle.ALTERNATIVE: "Alternative not considered: ",
    Angle.ASSUMPTION: "Unstated assumption: ",
    Angle.FAILURE_MODE: "Unhandled failure mode: ",
}
_NO_EVIDENCE_NOTE = " (No retrieved source supports this; treat it as a point to consider.)"
_SUGGESTION = "Acknowledge and answer this objection, or narrow the claim."


def rebuttal_severity(rebuttal: Rebuttal) -> Severity:
    """Severity of the finding for ``rebuttal``.

    ``STRONG_REBUTTAL`` when it cites retrieved evidence, otherwise
    ``CONSIDER`` (evidence-free rebuttals are marked for downgrade).
    """
    return Severity.STRONG_REBUTTAL if rebuttal.has_evidence else Severity.CONSIDER


class _SearchUnavailable(ReviewDeskError):
    """The search provider failed; research stops but rebuttals may continue."""


@dataclass
class _Gathered:
    label: str
    evidence: Evidence
    claim_id: str


@dataclass
class _Source:
    view: SourceView
    full_text: str
    retrieved_at: datetime | None = None


@dataclass
class _Session:
    """Per-call state: charges the shared meter and tracks this agent's usage."""

    ctx: ReviewContext
    usage: Usage = field(default_factory=Usage)
    started: float = field(default_factory=time.monotonic)
    updates: list[LedgerUpdate] = field(default_factory=list)
    gathered: list[_Gathered] = field(default_factory=list)

    def charge(self, usage: Usage) -> None:
        self.usage = self.usage + usage
        self.ctx.meter.charge(usage)

    async def llm(
        self, tag: str, messages: list[Message], schema: type[T], *, max_tokens: int = 1_500
    ) -> T:
        estimate = sum(len(m.content) for m in messages) // 4 + 1
        self.ctx.meter.require_tokens(estimate)
        response = await self.ctx.llm.complete(
            messages, schema=schema, model_tier=ModelTier.STRONG, tag=tag, max_tokens=max_tokens
        )
        self.charge(response.usage)
        if not isinstance(response.parsed, schema):
            raise SchemaError(f"{tag} returned no {schema.__name__}")
        return response.parsed

    async def search(self, query: str, k: int) -> list[SearchResult]:
        self.ctx.meter.require_search()
        try:
            hits = await self.ctx.search.search(query, k=k)
        except ProviderError as exc:
            raise _SearchUnavailable(f"search unavailable: {exc}") from exc
        finally:
            self.charge(Usage(search_calls=1))
        return hits

    async def fetch(self, url: str) -> FetchedPage | None:
        self.ctx.meter.require_time()
        try:
            return await self.ctx.fetcher.fetch(url)
        except (FetchError, ProviderError):
            return None
        finally:
            self.charge(Usage(fetch_calls=1))

    def step(
        self,
        claim_id: str,
        iteration: int,
        action: ResearchAction,
        input: str,
        summary: str = "",
        urls: list[str] | None = None,
    ) -> None:
        self.updates.append(
            AddResearchStep(
                claim_id=claim_id,
                step=ResearchStep(
                    agent=NAME,
                    iteration=max(1, min(iteration, MAX_RESEARCH_ITERATIONS)),
                    action=action,
                    input=input[:500],
                    summary=summary[:500],
                    evidence_urls=list(urls or []),
                ),
            )
        )

    def progress(self, message: str, percent: float) -> None:
        with contextlib.suppress(Exception):  # progress must never break the review
            self.ctx.emit_progress(
                ProgressEvent(step=NAME, message=message, percent=max(0.0, min(percent, 100.0)))
            )

    def final_usage(self) -> Usage:
        return self.usage + Usage(seconds=time.monotonic() - self.started)


def _clean_label(raw: str) -> str:
    return raw.strip().strip("[]()").strip().upper()


def _error_text(exc: Exception) -> str:
    return f"devil's advocate degraded: {type(exc).__name__}: {exc}"


class DevilsAdvocateAgent:
    """Devil's advocate: ``Agent`` and ``Debater``.

    ``max_claims`` caps how many thesis/supporting claims get researched,
    ``max_rebuttals`` how many rebuttals are kept, ``search_k`` the hits per
    search, and ``snippet_sources`` how many extra hits (snippet only) are
    shown to the model alongside the fetched page.
    """

    def __init__(
        self,
        *,
        max_claims: int = 5,
        max_rebuttals: int = 6,
        search_k: int = 5,
        snippet_sources: int = 2,
    ) -> None:
        self.max_claims = max_claims
        self.max_rebuttals = max_rebuttals
        self.search_k = search_k
        self.snippet_sources = snippet_sources

    @property
    def name(self) -> str:
        """``AgentName.DEVILS_ADVOCATE``."""
        return NAME

    # ------------------------------------------------------------------ run

    async def run(self, ctx: ReviewContext) -> AgentResult:
        """Research counter-evidence and return rebuttals, findings and trails."""
        s = _Session(ctx)
        targets = self._targets(ctx)
        if not targets:
            return AgentResult(agent=NAME, usage=s.final_usage())

        error: str | None = None
        for index, claim in enumerate(targets):
            s.progress(
                f"Devil's advocate: researching claim {index + 1} of {len(targets)}",
                100.0 * index / (len(targets) + 1),
            )
            iterations, claim_budget = self._iterations_for(ctx, claim, len(targets) - index)
            try:
                await self._research(s, claim, iterations, claim_budget)
            except (BudgetExceeded, _SearchUnavailable) as exc:
                error = _error_text(exc)
                break
            except ProviderError as exc:
                return self._result(s, [], _error_text(exc))

        s.progress(
            "Devil's advocate: writing the counter-case", 100.0 * len(targets) / (1 + len(targets))
        )
        try:
            rebuttals = await self._rebut(s, targets)
        except (ProviderError, BudgetExceeded) as exc:
            return self._result(s, [], error or _error_text(exc))
        s.progress("Devil's advocate: done", 100.0)
        return self._result(s, rebuttals, error)

    def _result(
        self, s: _Session, rebuttals: list[tuple[Rebuttal, Angle]], error: str | None
    ) -> AgentResult:
        findings: list[Finding] = []
        updates: list[LedgerUpdate] = list(s.updates)
        for rebuttal, angle in rebuttals:
            updates.append(AddRebuttal(rebuttal=rebuttal))
            findings.append(self._finding(s.ctx, rebuttal, angle))
        return AgentResult(
            agent=NAME,
            findings=findings,
            ledger_updates=updates,
            usage=s.final_usage(),
            error=error,
        )

    def _targets(self, ctx: ReviewContext) -> list[Claim]:
        claims = ctx.ledger.claims(ClaimType.THESIS, ClaimType.SUPPORTING)
        if ctx.plan is not None and ctx.plan.deep_research_claim_ids:
            deep = set(ctx.plan.deep_research_claim_ids)
            claims = [c for c in claims if c.id in deep] + [c for c in claims if c.id not in deep]
        return claims[: self.max_claims]

    def _iterations_for(
        self, ctx: ReviewContext, claim: Claim, remaining: int
    ) -> tuple[int, ClaimBudget]:
        plan = ctx.plan
        budget = plan.budget_for(claim.id) if plan is not None else ClaimBudget()
        iterations = min(budget.max_iterations, MAX_RESEARCH_ITERATIONS)
        deep = plan.deep_research_claim_ids if plan is not None else []
        if deep and claim.id not in deep:
            iterations = 1  # not a deep-research claim: a single search
        if ctx.meter.searches_left() < remaining * iterations:
            iterations = 1  # budget running short: a single search per claim
        return iterations, budget

    # ------------------------------------------------------------- research

    async def _research(
        self, s: _Session, claim: Claim, iterations: int, budget: ClaimBudget
    ) -> None:
        ctx = s.ctx
        tokens_at_start = s.usage.total_tokens
        plan = await s.llm(
            prompts.TAG_PLAN,
            prompts.plan_messages(ctx.profile, ctx.document.text, claim, ctx.focus),
            ResearchPlan,
        )
        queries = deque(q.strip()[:300] for q in plan.queries[:3] if q.strip())
        if not queries:
            queries.append(claim.text[:300])
        terms = keywords(claim.text, *queries)
        seen: set[str] = set()
        known: set[str] = set()
        notes: list[str] = []
        evidence: list[Evidence] = []
        follow: str | None = None
        searches = 0
        stop_reason = "iteration cap reached"
        iteration = 1

        for iteration in range(1, iterations + 1):
            sources: list[_Source] = []
            if follow is not None:
                url, follow = follow, None
                seen.add(url)
                page = await s.fetch(url)
                s.step(
                    claim.id,
                    iteration,
                    ResearchAction.FETCH_PAGE,
                    url,
                    "followed link" if page else "follow-up fetch failed",
                    [url] if page else [],
                )
                if page is not None:
                    sources.append(self._page_source(s, claim, iteration, page, terms, known, 1))
            else:
                if searches >= budget.max_search_calls:
                    stop_reason = "claim search budget used"
                    break
                if not queries:
                    stop_reason = "no further queries"
                    break
                query = queries.popleft()
                hits = await s.search(query, self.search_k)
                searches += 1
                known.update(h.url for h in hits)
                s.step(
                    claim.id,
                    iteration,
                    ResearchAction.SEARCH,
                    query,
                    f"{len(hits)} results",
                    [h.url for h in hits],
                )
                fresh = [h for h in hits if h.url not in seen]
                if fresh:
                    top = fresh[0]
                    seen.add(top.url)
                    page = await s.fetch(top.url)
                    s.step(
                        claim.id,
                        iteration,
                        ResearchAction.FETCH_PAGE,
                        top.url,
                        "fetched" if page else "fetch failed; using search snippet",
                        [top.url],
                    )
                    if page is not None:
                        sources.append(
                            self._page_source(s, claim, iteration, page, terms, known, 1)
                        )
                    elif top.snippet:
                        sources.append(self._snippet_source(top, 1))
                    for extra in fresh[1 : 1 + self.snippet_sources]:
                        if extra.snippet:
                            sources.append(self._snippet_source(extra, len(sources) + 1))

            if not sources:
                notes.append(f"Iteration {iteration}: nothing retrieved.")
                s.step(claim.id, iteration, ResearchAction.ASSESS, claim.id, "nothing to assess")
                if not queries and follow is None:
                    stop_reason = "no sources found"
                    break
                continue

            assessment = await s.llm(
                prompts.TAG_ASSESS,
                prompts.assess_messages(ctx.profile, claim, [x.view for x in sources], notes),
                Assessment,
                max_tokens=1_200,
            )
            found = self._verified_evidence(assessment, sources, evidence)
            for item in found:
                evidence.append(item)
                s.gathered.append(_Gathered(f"E{len(s.gathered) + 1}", item, claim.id))
            summary = assessment.summary.strip() or f"{len(found)} counter-evidence excerpts"
            notes.append(f"Iteration {iteration}: {summary[:300]}")
            s.step(
                claim.id,
                iteration,
                ResearchAction.ASSESS,
                claim.id,
                summary,
                [e.url for e in found],
            )

            if any(e.is_primary for e in evidence):
                stop_reason = "primary source found"
                break
            if len({domain(e.url) for e in evidence}) >= 2:
                stop_reason = "two independent sources agree"
                break
            if s.usage.total_tokens - tokens_at_start >= budget.max_tokens:
                stop_reason = "claim token budget used"
                break
            if assessment.next_action is NextAction.STOP:
                stop_reason = "assessment chose to stop"
                break
            if assessment.next_action is NextAction.FOLLOW_LINK:
                url = (assessment.follow_url or "").strip()
                if url and url in known and url not in seen:
                    follow = url
                else:
                    notes.append("Ignored a follow_url that was not in the retrieved material.")
            if follow is None and assessment.next_query and assessment.next_query.strip():
                query = assessment.next_query.strip()[:300]
                queries.appendleft(query)
                terms |= keywords(query)

        s.step(
            claim.id,
            iteration,
            ResearchAction.STOP,
            claim.id,
            f"{stop_reason}; {len(evidence)} counter-evidence excerpts",
            [e.url for e in evidence],
        )

    def _page_source(
        self,
        s: _Session,
        claim: Claim,
        iteration: int,
        page: FetchedPage,
        terms: set[str],
        known: set[str],
        number: int,
    ) -> _Source:
        known.update(urls_in(page.text))
        passages = find_in_page(page.text, terms)
        s.step(
            claim.id,
            iteration,
            ResearchAction.FIND_IN_PAGE,
            page.final_url,
            f"{len(passages)} matching passages",
            [page.final_url],
        )
        view = SourceView(
            label=f"S{number}",
            url=page.final_url,
            title=page.title or page.final_url,
            text="\n[...]\n".join(passages),
        )
        return _Source(view=view, full_text=page.text, retrieved_at=page.fetched_at)

    @staticmethod
    def _snippet_source(hit: SearchResult, number: int) -> _Source:
        view = SourceView(label=f"S{number}", url=hit.url, title=hit.title, text=hit.snippet)
        return _Source(view=view, full_text=hit.snippet)

    @staticmethod
    def _verified_evidence(
        assessment: Assessment, sources: list[_Source], existing: list[Evidence]
    ) -> list[Evidence]:
        by_label = {x.view.label: x for x in sources}
        have = {(e.url, normalize(e.excerpt)) for e in existing}
        found: list[Evidence] = []
        for item in assessment.evidence:
            source = by_label.get(_clean_label(item.source_id))
            if source is None:
                continue
            excerpt = locate_excerpt(item.excerpt, source.full_text)
            if excerpt is None:
                continue  # not in the source: invented or misquoted
            key = (source.view.url, normalize(excerpt))
            if key in have:
                continue
            have.add(key)
            retrieved = source.retrieved_at
            found.append(
                Evidence(
                    url=source.view.url,
                    title=source.view.title,
                    excerpt=excerpt,
                    retrieved_at=retrieved if retrieved is not None else utcnow(),
                    is_primary=item.is_primary,
                )
            )
        return found

    # ---------------------------------------------------------------- rebut

    async def _rebut(self, s: _Session, targets: list[Claim]) -> list[tuple[Rebuttal, Angle]]:
        ctx = s.ctx
        target_ids = {c.id for c in targets}
        context_lines = []
        for entry in ctx.ledger.entries.values():
            if entry.claim.id in target_ids:
                continue
            note = f" ({entry.verdict_note})" if entry.verdict_note else ""
            context_lines.append(
                f"[{entry.claim.id}] ({entry.claim.type.value}) "
                f"verdict={entry.verdict.value}{note}: {entry.claim.text}"
            )
        views = [EvidenceView(g.label, g.evidence, g.claim_id) for g in s.gathered]
        reply = await s.llm(
            prompts.TAG_REBUT,
            prompts.rebut_messages(
                ctx.profile, ctx.document.text, targets, context_lines, views, ctx.focus
            ),
            RebuttalSet,
            max_tokens=3_000,
        )
        built = [r for r in (self._build(ctx, d, s.gathered) for d in reply.rebuttals) if r]
        built.sort(key=lambda pair: -pair[0].strength)
        return built[: self.max_rebuttals]

    def _build(
        self, ctx: ReviewContext, draft: RebuttalDraft, gathered: list[_Gathered]
    ) -> tuple[Rebuttal, Angle] | None:
        argument = draft.argument.strip()
        if not argument:
            return None
        by_label = {g.label: g for g in gathered}
        cited: list[_Gathered] = []
        for raw in draft.evidence_ids:
            g = by_label.get(_clean_label(raw))
            if g is not None and g not in cited:
                cited.append(g)
        target_ids = self.repair_targets(ctx, draft.target_claim_ids)
        if not target_ids:
            target_ids = list(dict.fromkeys(g.claim_id for g in cited))
        if not target_ids:
            return None  # cannot tie it to any real claim: drop it
        rebuttal = Rebuttal(
            id=new_id("reb"),
            target_claim_ids=target_ids,
            argument=argument,
            evidence=[g.evidence for g in cited],
            strength=max(1, min(5, draft.strength)),
            target=draft.target,
        )
        return rebuttal, draft.angle

    @staticmethod
    def repair_targets(ctx: ReviewContext, raw_ids: list[str]) -> list[str]:
        """Map model-proposed claim IDs to real ledger IDs; drop the rest.

        Accepts exact IDs, IDs differing in case or wrapped in brackets, and
        the claim's exact text instead of its ID.
        """
        entries = ctx.ledger.entries
        by_lower = {cid.lower(): cid for cid in entries}
        by_text = {normalize(e.claim.text): cid for cid, e in entries.items()}
        out: list[str] = []
        for raw in raw_ids:
            cleaned = raw.strip().strip("[]()").strip()
            cid = (
                cleaned
                if cleaned in entries
                else by_lower.get(cleaned.lower()) or by_text.get(normalize(cleaned))
            )
            if cid is not None and cid not in out:
                out.append(cid)
        return out

    def _finding(self, ctx: ReviewContext, rebuttal: Rebuttal, angle: Angle) -> Finding:
        claims = [ctx.ledger.get(cid).claim for cid in rebuttal.target_claim_ids]
        anchor = max(claims, key=lambda c: c.importance)
        message = _ANGLE_PREFIX[angle] + rebuttal.argument
        if not rebuttal.has_evidence:
            message += _NO_EVIDENCE_NOTE
        return Finding(
            id=f"find_{rebuttal.id}",
            agent=NAME,
            severity=rebuttal_severity(rebuttal),
            span=anchor.span,
            message=message,
            evidence=list(rebuttal.evidence),
            suggestion=_SUGGESTION,
            claim_ids=list(rebuttal.target_claim_ids),
        )

    # --------------------------------------------------------------- debate

    async def respond(
        self, ctx: ReviewContext, claim_id: str, opposing_evidence: list[Evidence]
    ) -> DebatePosition:
        """One debate response to the fact-checker's evidence on ``claim_id``.

        Uses the rebuttals already on the claim's ledger entry. Never raises:
        on an unknown claim, provider error or exhausted budget it returns a
        position explaining that it could not respond (``concedes=False``).
        """
        entry = ctx.ledger.entries.get(claim_id)
        if entry is None:
            return DebatePosition(
                agent=NAME, claim_id=claim_id, argument="No such claim; no position to defend."
            )
        own: list[EvidenceView] = []
        for rebuttal in entry.rebuttals:
            for ev in rebuttal.evidence:
                if all(ev != v.evidence for v in own):
                    own.append(EvidenceView(f"E{len(own) + 1}", ev))
        opposing = [EvidenceView(f"O{i + 1}", ev) for i, ev in enumerate(opposing_evidence)]
        verdict = entry.verdict.value + (f" ({entry.verdict_note})" if entry.verdict_note else "")
        s = _Session(ctx)
        try:
            reply = await s.llm(
                prompts.TAG_DEBATE,
                prompts.debate_messages(
                    ctx.profile,
                    entry.claim,
                    verdict,
                    [r.argument for r in entry.rebuttals],
                    own,
                    opposing,
                ),
                DebateReply,
                max_tokens=800,
            )
        except (ProviderError, BudgetExceeded) as exc:
            return DebatePosition(
                agent=NAME,
                claim_id=claim_id,
                argument=f"The devil's advocate could not respond ({type(exc).__name__}).",
            )
        by_label = {v.label: v.evidence for v in own}
        cited: list[Evidence] = []
        for raw in reply.evidence_ids:
            chosen = by_label.get(_clean_label(raw))
            if chosen is not None and chosen not in cited:
                cited.append(chosen)
        argument = reply.argument.strip() or (
            "Concedes the point." if reply.concedes else "The rebuttal stands."
        )
        return DebatePosition(
            agent=NAME,
            claim_id=claim_id,
            argument=argument,
            evidence=cited,
            concedes=reply.concedes,
        )
