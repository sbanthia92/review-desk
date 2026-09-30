"""OriginalityAgent tests with FakeSearch and FakeFetcher (offline)."""

from __future__ import annotations

import pytest

from reviewdesk.agents.originality import OriginalityAgent
from reviewdesk.contracts import (
    Agent,
    AgentName,
    AgentResult,
    Budget,
    Document,
    FetchedPage,
    Finding,
    ProgressStep,
    ProviderError,
    RateLimitError,
    SearchResult,
    Severity,
)
from reviewdesk.testing.fakes import (
    FIXED_TIME,
    OPINION_DOC,
    FakeFetcher,
    FakeLLM,
    FakeSearch,
    ProgressRecorder,
    make_context,
    sample_documents,
)

LIVERPOOL = (
    "Liverpool won the 2019-20 Premier League with 99 points, and their press created "
    "more goals than any other team that season."
)
PRESSING = "High pressing is the single most important tactic in modern football."
SHOTS = "Teams that press high concede fewer shots, because opponents never get time on the ball."

COPY_URL = "https://copycat.example.com/blog/pressing"
OWN_URL = "https://myblog.example.com/posts/high-pressing"


def hit(url: str, snippet: str, *, title: str = "Some page", rank: int = 1) -> SearchResult:
    return SearchResult(url=url, title=title, snippet=snippet, rank=rank)


def assert_valid(result: AgentResult, doc: Document) -> None:
    """Invariants every originality result must satisfy."""
    assert result.agent == AgentName.ORIGINALITY
    assert result.ledger_updates == []
    for finding in result.findings:
        assert finding.heuristic is True
        assert finding.agent == AgentName.ORIGINALITY
        assert finding.severity == Severity.CONSIDER
        assert finding.span is not None
        assert finding.span.is_valid_for(doc.text)
        assert finding.evidence, "a possible match must cite at least one URL"
        assert "similarity" in finding.message
    for note in result.notes:
        for sentence in (LIVERPOOL, PRESSING, SHOTS):
            assert sentence not in note


def finding_for(result: AgentResult, doc: Document, sentence: str) -> Finding:
    matches = [
        f for f in result.findings if f.span is not None and f.span.text_of(doc.text) == sentence
    ]
    assert len(matches) == 1
    return matches[0]


def test_is_an_agent() -> None:
    agent = OriginalityAgent()
    assert isinstance(agent, Agent)
    assert agent.name == AgentName.ORIGINALITY


def test_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        OriginalityAgent(threshold=0.0)
    with pytest.raises(ValueError):
        OriginalityAgent(sample_size=-1)
    with pytest.raises(ValueError):
        OriginalityAgent(threshold=0.5, fetch_floor=0.7)


async def test_match_reports_url_span_and_score() -> None:
    search = FakeSearch(
        {
            "Liverpool won the 2019-20": [
                hit(
                    COPY_URL,
                    "Fan blog: Liverpool won the 2019-20 Premier League with 99 points, and "
                    "their press created more goals than any other team that season, it says.",
                    title="Pressing explained",
                )
            ]
        }
    )
    llm = FakeLLM()  # unscripted: any LLM call would fail the test
    ctx = make_context(OPINION_DOC, search=search, llm=llm)
    result = await OriginalityAgent().run(ctx)

    assert result.error is None
    assert_valid(result, OPINION_DOC)
    finding = finding_for(result, OPINION_DOC, LIVERPOOL)
    assert finding.evidence[0].url == COPY_URL
    assert finding.evidence[0].title == "Pressing explained"
    assert "Liverpool won the 2019-20 Premier League" in finding.evidence[0].excerpt
    assert "similarity 1.00" in finding.message
    assert COPY_URL in finding.message
    assert "Heuristic" in finding.message
    assert llm.calls == []
    # Exact-phrase query for the most distinctive sentence goes first.
    assert search.queries[0].startswith('"Liverpool won the 2019-20')


async def test_no_match_returns_no_findings() -> None:
    search = FakeSearch(
        default=[
            hit("https://example.org/a", "Arsenal's pressing numbers improved last season."),
            hit("https://example.org/b", "A history of the offside rule in English football."),
        ]
    )
    ctx = make_context(OPINION_DOC, search=search)
    result = await OriginalityAgent().run(ctx)

    assert result.error is None
    assert result.findings == []
    assert_valid(result, OPINION_DOC)
    assert len(search.queries) == 5
    assert result.usage.search_calls == 5


async def test_near_match_under_threshold_is_ignored() -> None:
    # Shares one short run with the sentence: related, but not a near-copy.
    search = FakeSearch(
        {
            "High pressing is the single": [
                hit(COPY_URL, "Is high pressing the key? Most important tactic debates rage on.")
            ]
        }
    )
    ctx = make_context(OPINION_DOC, search=search)
    result = await OriginalityAgent(fetch_pages=False).run(ctx)
    assert result.findings == []

    # The same snippet counts once the threshold is lowered.
    ctx = make_context(OPINION_DOC, search=search)
    loose = await OriginalityAgent(threshold=0.25, fetch_pages=False).run(ctx)
    assert_valid(loose, OPINION_DOC)
    assert finding_for(loose, OPINION_DOC, PRESSING)


async def test_light_edits_still_match() -> None:
    search = FakeSearch(
        {
            "High pressing is the single": [
                hit(COPY_URL, "High pressing is the single most important tactic in football.")
            ]
        }
    )
    ctx = make_context(OPINION_DOC, search=search)
    result = await OriginalityAgent().run(ctx)
    finding = finding_for(result, OPINION_DOC, PRESSING)
    assert "close match" in finding.message


async def test_own_url_results_are_skipped() -> None:
    doc = Document.from_text(OPINION_DOC.text, id="doc_own", source_url=OWN_URL)
    search = FakeSearch(
        {
            "Liverpool won the 2019-20": [
                hit("http://www.myblog.example.com/posts/high-pressing/", LIVERPOOL, rank=1),
                hit(COPY_URL, LIVERPOOL, rank=2),
            ],
            "Teams that press high": [hit(OWN_URL + "#comments", SHOTS)],
        }
    )
    ctx = make_context(doc, search=search)
    result = await OriginalityAgent().run(ctx)

    assert_valid(result, doc)
    finding = finding_for(result, doc, LIVERPOOL)
    assert [e.url for e in finding.evidence] == [COPY_URL]
    assert all(f.span is None or f.span.text_of(doc.text) != SHOTS for f in result.findings)
    assert any("own URL" in n for n in result.notes)


async def test_multiple_sources_cited_best_first() -> None:
    search = FakeSearch(
        {
            "Liverpool won the 2019-20": [
                hit("https://b.example.com/x", LIVERPOOL[:80]),
                hit("https://a.example.com/y", LIVERPOOL),
                hit("https://a.example.com/y/", LIVERPOOL),  # duplicate URL
            ]
        }
    )
    ctx = make_context(OPINION_DOC, search=search)
    result = await OriginalityAgent().run(ctx)
    finding = finding_for(result, OPINION_DOC, LIVERPOOL)
    assert [e.url for e in finding.evidence] == [
        "https://a.example.com/y",
        "https://b.example.com/x",
    ]
    assert "1 other source" in finding.message


async def test_suggestive_snippet_triggers_fetch_and_page_match() -> None:
    url = "https://copycat.example.com/full"
    search = FakeSearch(
        {"Teams that press high": [hit(url, "Teams that press high concede fewer ... read on")]}
    )
    page_text = "Intro paragraph about football tactics. " * 50 + SHOTS + " More filler text."
    fetcher = FakeFetcher(
        {
            url: FetchedPage(
                url=url,
                final_url=url,
                status=200,
                title="Copied article",
                text=page_text,
                fetched_at=FIXED_TIME,
            )
        }
    )
    ctx = make_context(OPINION_DOC, search=search, fetcher=fetcher)
    result = await OriginalityAgent().run(ctx)

    assert fetcher.fetched == [url]
    assert result.usage.fetch_calls == 1
    assert ctx.meter.used.fetch_calls == 1
    finding = finding_for(result, OPINION_DOC, SHOTS)
    assert finding.evidence[0].title == "Copied article"
    assert "concede fewer shots" in finding.evidence[0].excerpt
    assert_valid(result, OPINION_DOC)


async def test_fetch_disabled_and_fetch_failure() -> None:
    url = "https://copycat.example.com/full"
    search = FakeSearch(
        {"Teams that press high": [hit(url, "Teams that press high concede fewer ... read on")]}
    )
    fetcher = FakeFetcher()  # every URL unreachable
    ctx = make_context(OPINION_DOC, search=search, fetcher=fetcher)
    no_fetch = await OriginalityAgent(fetch_pages=False).run(ctx)
    assert fetcher.fetched == []
    assert no_fetch.findings == []

    ctx = make_context(OPINION_DOC, search=search, fetcher=fetcher)
    failed = await OriginalityAgent().run(ctx)
    assert fetcher.fetched == [url]
    assert failed.error is None
    assert failed.findings == []
    assert any("could not be fetched" in n for n in failed.notes)


async def test_fetched_page_redirecting_to_own_url_is_skipped() -> None:
    doc = Document.from_text(OPINION_DOC.text, id="doc_own", source_url=OWN_URL)
    url = "https://short.example.com/abc"
    search = FakeSearch(
        {"Teams that press high": [hit(url, "Teams that press high concede fewer ... read on")]}
    )
    fetcher = FakeFetcher({url: FetchedPage(url=url, final_url=OWN_URL, status=200, text=SHOTS)})
    ctx = make_context(doc, search=search, fetcher=fetcher)
    result = await OriginalityAgent().run(ctx)
    assert result.findings == []
    assert any("own URL" in n for n in result.notes)


async def test_sample_capped_by_search_budget() -> None:
    search = FakeSearch(default=[hit(COPY_URL, LIVERPOOL)])
    ctx = make_context(OPINION_DOC, search=search, budget=Budget(max_search_calls=2))
    result = await OriginalityAgent(sample_size=5).run(ctx)

    assert len(search.queries) == 2
    assert ctx.meter.used.search_calls == 2
    assert result.usage.search_calls == 2
    assert result.error is None
    assert any("3 sampled sentence(s) not checked" in n for n in result.notes)
    assert_valid(result, OPINION_DOC)


async def test_sample_size_config() -> None:
    search = FakeSearch()
    ctx = make_context(OPINION_DOC, search=search)
    result = await OriginalityAgent(sample_size=1).run(ctx)
    assert len(search.queries) == 1
    assert result.notes == []


async def test_budget_already_exhausted() -> None:
    search = FakeSearch()
    ctx = make_context(OPINION_DOC, search=search, budget=Budget(max_search_calls=0))
    result = await OriginalityAgent().run(ctx)
    assert search.queries == []
    assert result.findings == []
    assert result.error is not None and "budget" in result.error


async def test_time_budget_exhausted_stops_gracefully() -> None:
    search = FakeSearch()
    ctx = make_context(OPINION_DOC, search=search, budget=Budget(max_seconds=0))
    result = await OriginalityAgent().run(ctx)
    assert search.queries == []
    assert result.error is not None and "budget exhausted" in result.error


async def test_budget_spent_by_another_agent_mid_run() -> None:
    class DrainingSearch(FakeSearch):
        """Simulates a parallel agent using up the shared search budget."""

        async def search(self, query: str, *, k: int = 5) -> list[SearchResult]:
            results = await super().search(query, k=k)
            ctx.meter.used = ctx.meter.used.model_copy(update={"search_calls": 10})
            return results

    search = DrainingSearch(default=[hit(COPY_URL, LIVERPOOL)])
    ctx = make_context(OPINION_DOC, search=search, budget=Budget(max_search_calls=10))
    result = await OriginalityAgent().run(ctx)
    assert len(search.queries) == 1
    assert len(result.findings) == 1  # partial result kept
    assert result.error is not None and "budget exhausted" in result.error


async def test_search_error_returns_error_without_raising() -> None:
    search = FakeSearch(error=RateLimitError("search rate limited"))
    ctx = make_context(OPINION_DOC, search=search)
    result = await OriginalityAgent().run(ctx)
    assert result.findings == []
    assert result.error is not None and "search failed" in result.error
    assert len(search.queries) == 1
    assert ctx.meter.used.search_calls == 1


async def test_search_error_mid_run_keeps_partial_findings() -> None:
    class FlakySearch(FakeSearch):
        async def search(self, query: str, *, k: int = 5) -> list[SearchResult]:
            if self.queries:
                self.queries.append(query)
                raise ProviderError("search provider down")
            return await super().search(query, k=k)

    search = FlakySearch(default=[hit(COPY_URL, LIVERPOOL)])
    ctx = make_context(OPINION_DOC, search=search)
    result = await OriginalityAgent().run(ctx)
    assert len(result.findings) == 1
    assert result.error is not None
    assert any("not checked" in n for n in result.notes)
    assert_valid(result, OPINION_DOC)


async def test_progress_events() -> None:
    recorder = ProgressRecorder()
    ctx = make_context(OPINION_DOC, search=FakeSearch(), emit_progress=recorder)
    await OriginalityAgent(sample_size=2).run(ctx)
    assert set(recorder.steps) == {ProgressStep.REVIEWING}
    messages = [e.message for e in recorder.events]
    assert "Checking originality of sentence 2 of 2" in messages
    assert recorder.events[-1].percent == 100.0


async def test_progress_callback_errors_are_swallowed() -> None:
    def boom(_: object) -> None:
        raise RuntimeError("progress sink down")

    ctx = make_context(OPINION_DOC, search=FakeSearch(), emit_progress=boom)
    result = await OriginalityAgent().run(ctx)
    assert result.error is None


@pytest.mark.parametrize("doc", sample_documents(), ids=lambda d: d.id)
async def test_every_sample_doc_yields_valid_heuristic_findings(doc: Document) -> None:
    # Every search returns a verbatim copy of whatever sentence was queried.
    class EchoSearch(FakeSearch):
        async def search(self, query: str, *, k: int = 5) -> list[SearchResult]:
            self.queries.append(query)
            return [hit(COPY_URL, query.strip('"'))]

    search = EchoSearch()
    ctx = make_context(doc, search=search)
    result = await OriginalityAgent().run(ctx)
    assert result.error is None
    assert len(result.findings) == len(search.queries) > 0
    assert_valid(result, doc)
    for finding in result.findings:
        assert finding.span is not None
        assert finding.span.text_of(doc.text).strip() == finding.span.text_of(doc.text)
