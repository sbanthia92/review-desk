"""The originality checker agent.

Deterministic-first: sentences are sampled in code (``sentences``), each is
searched as an exact phrase, and search snippets (optionally the fetched page
of the most promising hit) are compared to the sentence in code
(``similarity``). No LLM is called. Every finding is ``heuristic=True`` with
``Severity.CONSIDER``: a match is a prompt to check attribution, not a verdict.

Search results and pages are untrusted data: they are only compared and
quoted as excerpts, never interpreted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from reviewdesk.agents.originality.sentences import Sentence, distinctive_sentences
from reviewdesk.agents.originality.similarity import Match, same_url, similarity
from reviewdesk.contracts.errors import BudgetExceeded, FetchError, ProviderError
from reviewdesk.contracts.interfaces import ReviewContext, SearchResult
from reviewdesk.contracts.models import (
    AgentName,
    AgentResult,
    Evidence,
    Finding,
    ProgressEvent,
    ProgressStep,
    Severity,
    Usage,
    utcnow,
)

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 5
"""Sentences searched per document (further capped by the search budget)."""

DEFAULT_THRESHOLD = 0.6
"""Minimum similarity for a possible match."""

FETCH_FLOOR_RATIO = 0.5
"""Default ``fetch_floor`` as a share of ``threshold``: snippets scoring at least
this (but under the threshold) may trigger a page fetch."""

DEFAULT_RESULTS_PER_QUERY = 5
MAX_QUERY_WORDS = 32
MAX_EVIDENCE = 3

_SEARCH_USAGE = Usage(search_calls=1)
_FETCH_USAGE = Usage(fetch_calls=1)


@dataclass
class _Hit:
    url: str
    title: str
    match: Match


@dataclass
class _Run:
    findings: list[Finding] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    checked: int = 0
    own_skipped: int = 0
    fetch_failed: int = 0
    error: str | None = None


def build_query(sentence: Sentence) -> str:
    """Exact-phrase query for a sentence (first ``MAX_QUERY_WORDS`` words)."""
    tokens = sentence.text.split()
    phrase = " ".join(tokens[:MAX_QUERY_WORDS]).strip(" \"'“”‘’.,;:!?")
    phrase = phrase.replace('"', "").replace("“", "").replace("”", "")
    return f'"{phrase}"'


class OriginalityAgent:
    """Searches the web for near-copies of the document's most distinctive sentences.

    Implements ``Agent`` with ``name == AgentName.ORIGINALITY``. Returns
    findings only (no ledger updates). ``sample_size`` caps the sentences
    searched, and never more than the searches left in the job budget.
    With ``fetch_pages``, a hit whose snippet is suggestive (at least
    ``fetch_floor``, default half the threshold) but under ``threshold`` is
    fetched, once per sentence, and compared against the page text. Never
    raises for provider or budget failures: returns partial findings with ``AgentResult.error`` set.
    """

    def __init__(
        self,
        *,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        threshold: float = DEFAULT_THRESHOLD,
        results_per_query: int = DEFAULT_RESULTS_PER_QUERY,
        fetch_pages: bool = True,
        fetch_floor: float | None = None,
    ) -> None:
        if sample_size < 0 or results_per_query <= 0:
            raise ValueError("sample_size must be >= 0 and results_per_query > 0")
        if fetch_floor is None:
            fetch_floor = threshold * FETCH_FLOOR_RATIO
        if not 0.0 < threshold <= 1.0 or not 0.0 <= fetch_floor <= threshold:
            raise ValueError("need 0 < threshold <= 1 and 0 <= fetch_floor <= threshold")
        self.sample_size = sample_size
        self.threshold = threshold
        self.results_per_query = results_per_query
        self.fetch_pages = fetch_pages
        self.fetch_floor = fetch_floor

    @property
    def name(self) -> str:
        """``AgentName.ORIGINALITY``."""
        return AgentName.ORIGINALITY

    def sample(self, ctx: ReviewContext) -> tuple[list[Sentence], int]:
        """``(sentences to search, sentences wanted)``, most distinctive first.

        ``wanted`` is capped by ``sample_size``; the returned sample is further
        capped by the searches left in the job budget.
        """
        candidates = distinctive_sentences(ctx.document.text)
        wanted = min(self.sample_size, len(candidates))
        return candidates[: min(wanted, ctx.meter.searches_left())], wanted

    async def run(self, ctx: ReviewContext) -> AgentResult:
        """Sample, search and compare; return heuristic findings."""
        sample, wanted = self.sample(ctx)
        state = _Run()
        if wanted and not sample:
            state.error = "originality check skipped: search budget exhausted"
        self._emit(ctx, f"Checking originality of {len(sample)} sentence(s)", 0.0)
        for i, sentence in enumerate(sample):
            self._emit(
                ctx,
                f"Checking originality of sentence {i + 1} of {len(sample)}",
                100.0 * i / max(1, len(sample)),
            )
            if not await self._check(ctx, sentence, state):
                break
        notes = self._notes(state, wanted)
        self._emit(ctx, f"Originality: {len(state.findings)} possible match(es)", 100.0)
        log.info(
            "originality: %d checked, %d findings, error=%s",
            state.checked,
            len(state.findings),
            bool(state.error),
        )
        return AgentResult(
            agent=self.name,
            findings=state.findings,
            usage=state.usage,
            error=state.error,
            notes=notes,
        )

    async def _check(self, ctx: ReviewContext, sentence: Sentence, state: _Run) -> bool:
        """Check one sentence. Returns False when the run must stop."""
        try:
            ctx.meter.require_search()
        except BudgetExceeded as exc:
            state.error = f"budget exhausted: {exc}"
            return False
        try:
            results = await ctx.search.search(build_query(sentence), k=self.results_per_query)
        except ProviderError as exc:
            state.error = f"originality search failed: {exc}"
            return False
        finally:
            ctx.meter.charge(_SEARCH_USAGE)
            state.usage = state.usage + _SEARCH_USAGE
        state.checked += 1

        hits, to_fetch = self._score_results(ctx, sentence, results, state)
        if not hits and to_fetch is not None and self.fetch_pages:
            try:
                ctx.meter.require_time()
            except BudgetExceeded as exc:
                state.error = f"budget exhausted: {exc}"
                return False
            hit = await self._fetch_and_compare(ctx, sentence, to_fetch, state)
            if hit is not None:
                hits.append(hit)
        if hits:
            state.findings.append(self._finding(sentence, hits))
        return True

    def _score_results(
        self,
        ctx: ReviewContext,
        sentence: Sentence,
        results: list[SearchResult],
        state: _Run,
    ) -> tuple[list[_Hit], SearchResult | None]:
        hits: list[_Hit] = []
        seen: set[str] = set()
        best_near: tuple[float, SearchResult] | None = None
        for result in results:
            if same_url(result.url, ctx.document.source_url):
                state.own_skipped += 1
                continue
            if any(same_url(result.url, u) for u in seen):
                continue
            seen.add(result.url)
            match = max(
                similarity(sentence.text, result.snippet),
                similarity(sentence.text, result.title),
                key=lambda m: m.score,
            )
            if match.score >= self.threshold:
                hits.append(_Hit(result.url, result.title, match))
            elif match.score >= self.fetch_floor and (
                best_near is None or match.score > best_near[0]
            ):
                best_near = (match.score, result)
        return hits, (best_near[1] if best_near else None)

    async def _fetch_and_compare(
        self, ctx: ReviewContext, sentence: Sentence, result: SearchResult, state: _Run
    ) -> _Hit | None:
        try:
            page = await ctx.fetcher.fetch(result.url)
        except FetchError:
            state.fetch_failed += 1
            return None
        finally:
            ctx.meter.charge(_FETCH_USAGE)
            state.usage = state.usage + _FETCH_USAGE
        if same_url(page.final_url, ctx.document.source_url):
            state.own_skipped += 1
            return None
        match = similarity(sentence.text, page.text)
        if match.score < self.threshold:
            return None
        return _Hit(result.url, page.title or result.title, match)

    def _finding(self, sentence: Sentence, hits: list[_Hit]) -> Finding:
        hits = sorted(hits, key=lambda h: (-h.match.score, h.url))[:MAX_EVIDENCE]
        top = hits[0].match.score
        now = utcnow()
        kind = "near-verbatim" if top >= 0.95 else "close"
        others = f" and {len(hits) - 1} other source(s)" if len(hits) > 1 else ""
        return Finding(
            agent=self.name,
            severity=Severity.CONSIDER,
            span=sentence.span,
            message=(
                f"Possible {kind} match online (similarity {top:.2f}) at "
                f"{hits[0].url}{others}. Heuristic check, not proof of copying."
            ),
            evidence=[
                Evidence(url=h.url, title=h.title, excerpt=h.match.excerpt, retrieved_at=now)
                for h in hits
            ],
            suggestion="If this wording comes from the source, quote and cite it.",
            heuristic=True,
        )

    def _notes(self, state: _Run, wanted: int) -> list[str]:
        notes: list[str] = []
        unchecked = wanted - state.checked
        if unchecked > 0:
            notes.append(f"{unchecked} sampled sentence(s) not checked for originality")
        if state.own_skipped:
            notes.append(f"{state.own_skipped} result(s) from the document's own URL ignored")
        if state.fetch_failed:
            notes.append(f"{state.fetch_failed} page(s) could not be fetched")
        return notes

    def _emit(self, ctx: ReviewContext, message: str, percent: float) -> None:
        try:
            ctx.emit_progress(
                ProgressEvent(step=ProgressStep.REVIEWING, message=message, percent=percent)
            )
        except Exception:  # progress must never break the check
            log.warning("originality: progress callback raised", exc_info=True)
