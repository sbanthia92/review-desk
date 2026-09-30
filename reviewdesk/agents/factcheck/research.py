"""The bounded research loop for one factual claim.

Per iteration (hard cap ``MAX_RESEARCH_ITERATIONS``): search (or follow a
link to the primary source), fetch the most promising pages, find the
relevant passages, and let the model assess each source. The loop stops when
a primary source settles the claim, when two independent sources agree, when
the model says further research will not help, when the budget runs out, or
at the cap. A final judge call proposes the verdict; code enforces that
"verified" and "wrong" are only given when the evidence settles the claim,
otherwise the claim is "unsupported" with the confidence noted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

from reviewdesk.agents.factcheck import prompts
from reviewdesk.agents.factcheck.schemas import (
    Assessment,
    Judgment,
    NextStep,
    QueryPlan,
    Stance,
)
from reviewdesk.agents.factcheck.sources import (
    Source,
    find_in_page,
    is_http_url,
    looks_like_injection,
    primary_hint,
    relevant,
    settles,
    verified_quote,
)
from reviewdesk.contracts import (
    MAX_RESEARCH_ITERATIONS,
    AddResearchStep,
    AgentName,
    BudgetExceeded,
    Claim,
    ClaimBudget,
    Evidence,
    FetchedPage,
    FetchError,
    Finding,
    LedgerUpdate,
    Message,
    ModelTier,
    ProviderError,
    ResearchAction,
    ResearchStep,
    ReviewContext,
    SchemaError,
    SearchResult,
    SetVerdict,
    Severity,
    Usage,
    Verdict,
    utcnow,
)

FETCHES_PER_ITERATION = 2
"""Pages fetched per search (primary-looking URLs first)."""
SNIPPET_SOURCES = 2
"""Extra search results shown to the model as snippets only."""
MAX_QUERIES = 3
MAX_COUNTER_EVIDENCE = 3
SEARCH_K = 5
MAX_NOTE_CHARS = 500

_OUT_QUERIES = 300
_OUT_ASSESS = 800
_OUT_JUDGE = 600

_T = TypeVar("_T", bound=BaseModel)


def estimate_tokens(messages: list[Message], max_output: int) -> int:
    """Rough token estimate for a call (4 characters per token plus output)."""
    return sum(len(m.content) for m in messages) // 4 + max_output


def _clip(text: str, limit: int = MAX_NOTE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class ClaimOutcome:
    """What researching one claim produced."""

    claim_id: str
    updates: list[LedgerUpdate] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    verdict: Verdict | None = None
    error: str | None = None


class ClaimResearch:
    """Researches one claim within its budget. Use once per claim.

    ``deep`` claims get the full loop; others get a single search and one
    iteration. Raises ``ProviderError`` for provider failures (the caller
    degrades); budget exhaustion is handled here.
    """

    def __init__(
        self, ctx: ReviewContext, claim: Claim, budget: ClaimBudget, *, deep: bool
    ) -> None:
        self.ctx = ctx
        self.claim = claim
        self.deep = deep
        cap = min(budget.max_iterations, MAX_RESEARCH_ITERATIONS)
        self.max_iterations = cap if deep else 1
        self.max_searches = budget.max_search_calls if deep else min(1, budget.max_search_calls)
        self.max_tokens = budget.max_tokens
        self.usage = Usage()
        self.tokens_used = 0
        self.searches_used = 0
        self.actions = 0
        self.iteration = 1
        self.sources: dict[str, Source] = {}
        self.outcome = ClaimOutcome(claim_id=claim.id)

    # -- metered tools -----------------------------------------------------

    def _charge(self, usage: Usage) -> None:
        self.ctx.meter.charge(usage)
        self.usage = self.usage + usage
        self.tokens_used += usage.total_tokens

    async def _llm(
        self, messages: list[Message], schema: type[_T], tag: str, max_output: int
    ) -> _T:
        estimate = estimate_tokens(messages, max_output)
        if self.tokens_used + estimate > self.max_tokens:
            raise BudgetExceeded("claim token budget exhausted")
        self.ctx.meter.require_tokens(estimate)
        response = await self.ctx.llm.complete(
            messages, schema=schema, model_tier=ModelTier.MID, tag=tag, max_tokens=max_output
        )
        self._charge(response.usage)
        if not isinstance(response.parsed, schema):
            raise SchemaError(f"{tag} reply did not match {schema.__name__}")
        return response.parsed

    async def _search(self, query: str) -> list[SearchResult]:
        if self.searches_used >= self.max_searches:
            raise BudgetExceeded("claim search budget exhausted")
        self.ctx.meter.require_search()
        # Charge before awaiting so concurrent claims cannot overspend the job.
        self.searches_used += 1
        self._charge(Usage(search_calls=1))
        return await self.ctx.search.search(query, k=SEARCH_K)

    async def _fetch(self, url: str) -> FetchedPage | None:
        self.ctx.meter.require_time()
        self._charge(Usage(fetch_calls=1))
        try:
            return await self.ctx.fetcher.fetch(url)
        except FetchError:
            return None

    def _step(
        self,
        action: ResearchAction,
        input: str,
        summary: str = "",
        urls: list[str] | None = None,
    ) -> None:
        step = ResearchStep(
            agent=AgentName.FACTCHECK,
            iteration=min(self.iteration, MAX_RESEARCH_ITERATIONS),
            action=action,
            input=_clip(input, 300),
            summary=_clip(summary, 300),
            evidence_urls=list(urls or []),
        )
        self.outcome.updates.append(AddResearchStep(claim_id=self.claim.id, step=step))

    # -- loop pieces -------------------------------------------------------

    async def _plan_queries(self) -> list[str]:
        if not self.deep:
            return [self.claim.text]
        plan = await self._llm(
            prompts.queries_messages(self.claim), QueryPlan, prompts.TAG_QUERIES, _OUT_QUERIES
        )
        queries = [" ".join(q.split()) for q in plan.queries if q.strip()]
        return queries[:MAX_QUERIES] or [self.claim.text]

    async def _read(
        self, url: str, *, query: str, title: str, snippet: str, kind: str = "page"
    ) -> Source | None:
        """Fetch a page, find the relevant passages and register the source."""
        page = await self._fetch(url)
        self.actions += 1
        if page is None:
            if not snippet:
                self._step(ResearchAction.FETCH_PAGE, url, "Fetch failed.")
                self.sources[url] = Source(url=url, title=title, text="", passages=[])
                return None
            self._step(ResearchAction.FETCH_PAGE, url, "Fetch failed; using the search snippet.")
            text, passages, kind, can_be_primary = snippet, [snippet], "snippet", False
        else:
            self._step(ResearchAction.FETCH_PAGE, url, f"Fetched: {page.title}", [page.final_url])
            title = page.title or title
            text, can_be_primary = page.text, True
            passages = find_in_page(text, self.claim.text, query)
        source = Source(
            url=url,
            title=title or url,
            text=text,
            passages=passages,
            kind=kind,
            can_be_primary=can_be_primary,
            suspicious=looks_like_injection(text),
            retrieved_at=page.fetched_at if page is not None else utcnow(),
        )
        self.sources[url] = source
        if source.suspicious:
            self._step(
                ResearchAction.FIND_IN_PAGE,
                url,
                "Skipped: the page contains instructions aimed at the reviewer.",
            )
            return None
        self._step(ResearchAction.FIND_IN_PAGE, url, f"{len(passages)} relevant passage(s).")
        return source if passages else None

    async def _search_and_read(self, query: str) -> list[Source]:
        results = await self._search(query)
        self.actions += 1
        self._step(
            ResearchAction.SEARCH, query, f"{len(results)} result(s).", [r.url for r in results]
        )
        fresh = [r for r in results if is_http_url(r.url) and r.url not in self.sources]
        ranked = sorted(fresh, key=lambda r: (not primary_hint(r.url), r.rank))
        to_fetch, rest = ranked[:FETCHES_PER_ITERATION], ranked[FETCHES_PER_ITERATION:]
        batch: list[Source] = []
        for result in to_fetch:
            source = await self._read(
                result.url, query=query, title=result.title, snippet=result.snippet
            )
            if source is not None:
                batch.append(source)
        for result in rest[:SNIPPET_SOURCES]:
            if not result.snippet:
                continue
            source = Source(
                url=result.url,
                title=result.title or result.url,
                text=result.snippet,
                passages=[result.snippet],
                kind="snippet",
                can_be_primary=False,
                suspicious=looks_like_injection(result.snippet),
            )
            self.sources[result.url] = source
            if not source.suspicious:
                batch.append(source)
        return batch

    async def _assess(self, batch: list[Source], query: str) -> Assessment:
        assessment = await self._llm(
            prompts.assess_messages(self.claim, batch, query),
            Assessment,
            prompts.TAG_ASSESS,
            _OUT_ASSESS,
        )
        by_url = {s.url: s for s in batch}
        for judged in assessment.sources:
            source = by_url.get(judged.url)
            if source is None:  # the model may only judge what it was shown
                continue
            source.stance = judged.stance
            if judged.stance is Stance.IRRELEVANT:
                source.evidence = None
                continue
            fallback = source.passages[0] if source.passages else source.text
            source.evidence = Evidence(
                url=source.url,
                title=source.title,
                excerpt=verified_quote(judged.quote, source.text, fallback),
                retrieved_at=source.retrieved_at,
                is_primary=judged.is_primary and source.can_be_primary,
            )
        used = [s.url for s in batch if s.evidence is not None]
        summary = assessment.summary or f"{len(used)} of {len(batch)} source(s) relevant."
        self._step(ResearchAction.ASSESS, query or "follow-up", summary, used)
        return assessment

    def _settled(self) -> bool:
        sources = list(self.sources.values())
        return settles(sources, Stance.SUPPORTS) or settles(sources, Stance.CONTRADICTS)

    # -- entry points ------------------------------------------------------

    async def run(self) -> ClaimOutcome:
        """Research the claim and return its verdict, findings and trail."""
        reason = "Research cap reached without a settling source."
        if self.max_searches == 0 or self.ctx.meter.searches_left() == 0:
            return await self._conclude("No search budget left.")
        try:
            pending = deque(await self._plan_queries())
            follow: str | None = None
            query = ""
            tried: set[str] = set()
            for iteration in range(1, self.max_iterations + 1):
                self.iteration = iteration
                if follow is not None:
                    source = await self._read(follow, query="", title=follow, snippet="")
                    batch = [source] if source is not None else []
                    follow = None
                elif pending:
                    query = pending.popleft()
                    tried.add(query.casefold())
                    batch = await self._search_and_read(query)
                else:
                    reason = "No further queries to try."
                    break
                if not batch:
                    continue
                assessment = await self._assess(batch, query)
                if self._settled():
                    reason = "Settled by the evidence."
                    break
                if assessment.next_step is NextStep.STOP:
                    reason = "Research stopped: no better sources expected."
                    break
                if (
                    assessment.next_step is NextStep.FOLLOW
                    and assessment.follow_url
                    and is_http_url(assessment.follow_url)
                    and assessment.follow_url not in self.sources
                ):
                    follow = assessment.follow_url
                elif assessment.next_query and assessment.next_query.strip():
                    pending.appendleft(" ".join(assessment.next_query.split()))
                while pending and pending[0].casefold() in tried:
                    pending.popleft()
        except BudgetExceeded as exc:
            reason = f"Budget ran short ({exc})."
        return await self._conclude(reason)

    async def recheck(self, prior: list[Evidence], counter: list[Evidence]) -> ClaimOutcome:
        """Re-judge the claim once, given the devil's advocate's counter-evidence.

        Counter-evidence pages are fetched so excerpts are re-validated; if a
        page cannot be fetched, its excerpt is used but it cannot count as a
        primary source. No new searches are made.
        """
        batch: list[Source] = []
        try:
            for ev in counter[:MAX_COUNTER_EVIDENCE]:
                if ev.url in self.sources:
                    continue
                source: Source | None = None
                if is_http_url(ev.url):
                    source = await self._read(
                        ev.url,
                        query=ev.excerpt,
                        title=ev.title,
                        snippet=ev.excerpt,
                        kind="counter-evidence",
                    )
                elif ev.excerpt:
                    source = Source(
                        url=ev.url,
                        title=ev.title,
                        text=ev.excerpt,
                        passages=[ev.excerpt],
                        kind="counter-evidence",
                        can_be_primary=False,
                        suspicious=looks_like_injection(ev.excerpt),
                    )
                    self.sources[ev.url] = source
                    source = None if source.suspicious else source
                if source is not None:
                    batch.append(source)
            for ev in prior:
                if ev.url in self.sources or not ev.excerpt:
                    continue
                source = Source(
                    url=ev.url,
                    title=ev.title,
                    text=ev.excerpt,
                    passages=[ev.excerpt],
                    kind="prior-evidence",
                    can_be_primary=ev.is_primary,
                    retrieved_at=ev.retrieved_at,
                )
                self.sources[ev.url] = source
                batch.append(source)
            if not batch:
                self._step(ResearchAction.STOP, "re-check", "No usable counter-evidence.")
                return self.outcome
            self.actions += 1
            await self._assess(batch, "re-check")
        except BudgetExceeded as exc:
            self._step(ResearchAction.STOP, "re-check", f"Re-check skipped: {exc}.")
            return self.outcome
        return await self._conclude("Re-checked against the devil's advocate's evidence.")

    # -- verdict -----------------------------------------------------------

    async def _conclude(self, reason: str) -> ClaimOutcome:
        sources = list(self.sources.values())
        supporting = relevant(sources, Stance.SUPPORTS)
        contradicting = relevant(sources, Stance.CONTRADICTS)
        if self.actions == 0:
            self._step(ResearchAction.STOP, "unchecked", reason)
            return self._finish(Verdict.UNCHECKED, [], None, f"Not checked. {reason}", None)

        judgment: Judgment | None = None
        if supporting or contradicting:
            try:
                judgment = await self._llm(
                    prompts.judge_messages(
                        self.claim,
                        [s.evidence for s in supporting if s.evidence],
                        [s.evidence for s in contradicting if s.evidence],
                    ),
                    Judgment,
                    prompts.TAG_JUDGE,
                    _OUT_JUDGE,
                )
            except BudgetExceeded:
                judgment = None
            except ProviderError as exc:  # decide from the evidence alone
                judgment = None
                self.outcome.error = f"judge failed: {type(exc).__name__}"

        sup_settled = settles(sources, Stance.SUPPORTS)
        con_settled = settles(sources, Stance.CONTRADICTS)
        if judgment is not None:
            proposed, confidence = judgment.verdict, judgment.confidence
            explanation = judgment.explanation
        elif sup_settled != con_settled:
            proposed = "verified" if sup_settled else "wrong"
            decisive = supporting if sup_settled else contradicting
            primary = any(s.evidence and s.evidence.is_primary for s in decisive)
            confidence = 0.8 if primary else 0.65
            explanation = "Decided from the evidence without a final judgment call."
        else:
            proposed, confidence, explanation = "unsupported", 0.3, ""

        def evidence_of(items: list[Source]) -> list[Evidence]:
            return [s.evidence for s in items if s.evidence is not None]

        if proposed == "verified" and sup_settled:
            self._step(ResearchAction.STOP, "verified", reason, [s.url for s in supporting])
            return self._finish(
                Verdict.VERIFIED, evidence_of(supporting), confidence, explanation, None
            )
        if proposed == "wrong" and con_settled:
            self._step(ResearchAction.STOP, "wrong", reason, [s.url for s in contradicting])
            correction = judgment.correction if judgment is not None else None
            return self._finish(
                Verdict.WRONG, evidence_of(contradicting), confidence, explanation, correction
            )

        note = reason
        if proposed != "unsupported":
            note = (
                f"{reason} The evidence leaned {proposed} (confidence {confidence:.2f}) but "
                "did not settle it: that needs a primary source or two independent sources."
            )
            confidence = min(confidence, 0.5)
        elif explanation:
            note = f"{reason} {explanation}"
        self._step(ResearchAction.STOP, "unsupported", note)
        evidence = evidence_of(supporting) + evidence_of(contradicting)
        evidence.sort(key=lambda e: not e.is_primary)
        return self._finish(Verdict.UNSUPPORTED, evidence, confidence, note, None)

    def fail(self, exc: BaseException) -> ClaimOutcome:
        """Degrade after a provider failure: keep the trail, mark the claim unchecked."""
        self.outcome.findings.clear()
        self.outcome.updates = [u for u in self.outcome.updates if not isinstance(u, SetVerdict)]
        self._step(ResearchAction.STOP, "error", f"Provider error ({type(exc).__name__}).")
        outcome = self._finish(
            Verdict.UNCHECKED, [], None, "Not checked: a provider error stopped the research.", None
        )
        message = " ".join(str(exc).split())[:200]
        outcome.error = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
        return outcome

    def _finish(
        self,
        verdict: Verdict,
        evidence: list[Evidence],
        confidence: float | None,
        note: str,
        correction: str | None,
    ) -> ClaimOutcome:
        note = _clip(note)
        self.outcome.updates.append(
            SetVerdict(
                claim_id=self.claim.id,
                verdict=verdict,
                evidence=evidence,
                confidence=confidence,
                note=note,
            )
        )
        if verdict is Verdict.WRONG:
            self.outcome.findings.append(
                Finding(
                    agent=AgentName.FACTCHECK,
                    severity=Severity.FACTUAL_ERROR,
                    span=self.claim.span,
                    message=_clip(f"Retrieved evidence contradicts this claim. {note}"),
                    evidence=evidence,
                    suggestion=_clip(correction) if correction else None,
                    claim_ids=[self.claim.id],
                )
            )
        elif verdict is Verdict.UNSUPPORTED:
            self.outcome.findings.append(
                Finding(
                    agent=AgentName.FACTCHECK,
                    severity=Severity.UNSUPPORTED,
                    span=self.claim.span,
                    message=_clip(f"No retrieved source settles this claim. {note}"),
                    evidence=evidence,
                    suggestion="Cite a primary source for this claim, or soften it.",
                    claim_ids=[self.claim.id],
                )
            )
        self.outcome.verdict = verdict
        self.outcome.usage = self.usage
        return self.outcome
