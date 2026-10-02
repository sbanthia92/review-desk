"""Verdict precision: the relaxed single-source rule, weak sources, and the second look."""

from __future__ import annotations

from typing import Any

from reviewdesk.agents.factcheck import FactCheckAgent
from reviewdesk.agents.factcheck.sources import is_low_quality, is_reputable
from reviewdesk.contracts import Message, Severity, Verdict
from reviewdesk.testing.fakes import FakeFetcher, FakeLLM, FakeSearch, make_context
from tests.agents.factcheck.test_agent import (
    GOALS,
    POINTS,
    assess,
    hit,
    judge,
    ledger_of,
    src,
    verdicts,
)

BBC = "https://www.bbc.com/sport/football/53000000"
BBC_TEXT = (
    "Liverpool won the 2019-20 Premier League with 99 points. Manchester City's press "
    "created more goals than any other team that season, 14 in total."
)
BLOG = "https://someblog.example.com/liverpool"
INSTA = "https://www.instagram.com/p/abc"
FACEBOOK = "https://www.facebook.com/fans/posts/1"
QUOTE = "Liverpool won the 2019-20 Premier League with 99 points."


def _ctx(llm: FakeLLM, search: FakeSearch, fetcher: FakeFetcher, *quotes: Any) -> Any:
    return make_context(ledger=ledger_of(*quotes), llm=llm, search=search, fetcher=fetcher)


def test_host_quality_lists():
    assert is_reputable("https://en.wikipedia.org/wiki/X")
    assert is_reputable("https://www.bbc.co.uk/news/1")
    assert is_reputable("https://data.census.gov/table")
    assert not is_reputable("https://notwikipedia.org/x")
    assert is_low_quality("https://m.facebook.com/x") and is_low_quality(INSTA)
    assert not is_low_quality(BBC)


async def test_one_reputable_source_verifies_an_uncontested_claim():
    llm = FakeLLM(
        {
            "factcheck.queries": {"queries": ["liverpool 99 points"]},
            "factcheck.assess": assess(src(BBC, "supports", QUOTE)),
            "factcheck.judge": judge("verified", 0.9),
        }
    )
    ctx = _ctx(
        llm, FakeSearch(default=[hit(BBC)]), FakeFetcher({BBC: BBC_TEXT}), (POINTS, 0.6, "c1")
    )
    result = await FactCheckAgent().run(ctx)
    verdict = verdicts(result)["c1"]
    assert verdict.verdict is Verdict.VERIFIED
    assert verdict.confidence == 0.7  # capped: one source, not two
    assert "one reputable source" in verdict.note
    assert result.findings == []


async def test_one_unknown_source_is_could_not_confirm_not_unsupported():
    llm = FakeLLM(
        {
            "factcheck.queries": {"queries": ["liverpool 99 points"]},
            "factcheck.assess": assess(src(BLOG, "supports", QUOTE)),
            "factcheck.judge": judge("verified", 0.9),
        }
    )
    ctx = _ctx(
        llm, FakeSearch(default=[hit(BLOG)]), FakeFetcher({BLOG: BBC_TEXT}), (POINTS, 0.6, "c1")
    )
    result = await FactCheckAgent().run(ctx)
    assert verdicts(result)["c1"].verdict is Verdict.UNSUPPORTED
    [finding] = result.findings
    assert finding.severity is Severity.CONSIDER
    assert finding.message.startswith("Could not confirm this claim")


async def test_two_social_posts_do_not_settle_a_claim():
    llm = FakeLLM(
        {
            "factcheck.queries": {"queries": ["liverpool 99 points"]},
            "factcheck.assess": assess(
                src(INSTA, "supports", QUOTE), src(FACEBOOK, "supports", QUOTE)
            ),
            "factcheck.judge": judge("verified", 0.9),
        }
    )
    ctx = _ctx(
        llm,
        FakeSearch(default=[hit(INSTA, rank=1), hit(FACEBOOK, rank=2)]),
        FakeFetcher({INSTA: BBC_TEXT, FACEBOOK: BBC_TEXT}),
        (POINTS, 0.6, "c1"),
    )
    result = await FactCheckAgent().run(ctx)
    assert verdicts(result)["c1"].verdict is Verdict.UNSUPPORTED
    assert result.findings[0].severity is Severity.CONSIDER


async def test_no_source_found_is_could_not_confirm():
    llm = FakeLLM({"factcheck.queries": {"queries": ["liverpool 99 points"]}})
    ctx = _ctx(llm, FakeSearch(default=[]), FakeFetcher(), (POINTS, 0.6, "c1"))
    result = await FactCheckAgent().run(ctx)
    verdict = verdicts(result)["c1"]
    assert verdict.verdict is Verdict.UNSUPPORTED
    assert "No relevant source was found" in verdict.note
    [finding] = result.findings
    assert finding.severity is Severity.CONSIDER
    assert "Cite a primary source" not in (finding.suggestion or "")


async def test_sources_that_do_not_back_the_claim_are_a_must_fix_without_contradictory_note():
    explanation = "The sources say 97 points in a different season, not 99."
    llm = FakeLLM(
        {
            "factcheck.queries": {"queries": ["liverpool 99 points"]},
            "factcheck.assess": assess(src(BBC, "contradicts", QUOTE)),
            "factcheck.judge": {
                "verdict": "unsupported",
                "confidence": 0.6,
                "explanation": explanation,
            },
        }
    )
    ctx = _ctx(
        llm, FakeSearch(default=[hit(BBC)]), FakeFetcher({BBC: BBC_TEXT}), (POINTS, 0.6, "c1")
    )
    result = await FactCheckAgent().run(ctx)
    verdict = verdicts(result)["c1"]
    assert verdict.verdict is Verdict.UNSUPPORTED
    assert verdict.note == explanation  # no "Settled by the evidence." prefix
    [finding] = result.findings
    assert finding.severity is Severity.UNSUPPORTED
    assert finding.message.startswith("Retrieved sources do not back this claim as written")


async def test_second_look_uses_pages_fetched_for_other_claims():
    def queries(messages: list[Message], schema: Any) -> dict[str, Any]:
        goals = 'claim_id="c_goals"' in messages[-1].content
        return {"queries": ["press goals" if goals else "liverpool 99 points"]}

    def assess_reply(messages: list[Message], schema: Any) -> dict[str, Any]:
        if 'claim_id="c_goals"' in messages[-1].content:
            quote = (
                "Manchester City's press created more goals than any other team that "
                "season, 14 in total."
            )
            return assess(src(BBC, "contradicts", quote))
        return assess(src(BBC, "supports", QUOTE))

    def judge_reply(messages: list[Message], schema: Any) -> dict[str, Any]:
        if 'claim_id="c_goals"' in messages[-1].content:
            return judge("wrong", 0.85, correction="Manchester City led this, not Liverpool.")
        return judge("verified", 0.9)

    llm = FakeLLM(
        {
            "factcheck.queries": queries,
            "factcheck.assess": assess_reply,
            "factcheck.judge": judge_reply,
        }
    )
    # Only the points claim's search finds anything; the goals claim finds nothing.
    search = FakeSearch({"99 points": [hit(BBC)]}, default=[])
    fetcher = FakeFetcher({BBC: BBC_TEXT})
    ctx = _ctx(llm, search, fetcher, (POINTS, 0.6, "c_points"), (GOALS, 0.8, "c_goals"))

    result = await FactCheckAgent().run(ctx)

    got = verdicts(result)
    assert got["c_points"].verdict is Verdict.VERIFIED
    goals = got["c_goals"]
    assert goals.verdict is Verdict.UNSUPPORTED
    assert [e.url for e in goals.evidence] == [BBC]  # found on the second look
    assert "One source contradicts this claim" in goals.note
    [finding] = result.findings
    assert finding.claim_ids == ["c_goals"]
    assert finding.severity is Severity.UNSUPPORTED
    # The second look costs no searches or fetches.
    assert len(search.queries) == 2
    assert fetcher.fetched == [BBC]
    assert result.usage.search_calls == 2 and result.usage.fetch_calls == 1
    # One verdict per claim, even though the goals claim was judged twice.
    assert (
        len(
            [
                u
                for u in result.ledger_updates
                if getattr(u, "claim_id", "") == "c_goals" and u.kind == "set_verdict"
            ]
        )
        == 1
    )


def test_surrounding_context_gives_title_and_paragraph():
    from reviewdesk.agents.factcheck import prompts
    from reviewdesk.agents.factcheck.research import MAX_CONTEXT_CHARS, surrounding_context
    from reviewdesk.contracts import ClaimType
    from reviewdesk.testing.fakes import OPINION_DOC, claim_for

    text = OPINION_DOC.text
    quote = "That is simply not true"
    start = text.index(quote)
    context = surrounding_context(text, start, start + len(quote))
    assert context.startswith("# Why high pressing wins titles")
    assert "Critics say pressing exhausts players by February." in context
    assert "Liverpool won the 2019-20" not in context  # a different paragraph

    long_text = "Title\n\n" + ("word " * 600) + "THE CLAIM " + ("word " * 600)
    start = long_text.index("THE CLAIM")
    clipped = surrounding_context(long_text, start, start + 9)
    assert "THE CLAIM" in clipped and len(clipped) <= MAX_CONTEXT_CHARS + 210

    claim = claim_for(OPINION_DOC, quote, ClaimType.FACTUAL, 0.5, id="c1")
    block = prompts.document_block(claim, context)
    assert "Surrounding passage (context only, do not fact-check it):" in block
    assert block.rstrip().endswith("</untrusted_document>")


def test_passage_matching_tolerates_spelling_variants():
    from reviewdesk.agents.factcheck.sources import find_in_page, shared_terms

    page = "The aircraft landed at Munich for refuelling before the flight home. Unrelated text."
    claim = "The plane took a planned refueling stop at Munich."
    assert find_in_page(page, claim) == [
        "The aircraft landed at Munich for refuelling before the flight home."
    ]
    assert shared_terms(claim, page) >= 2


# -- multi-part claims and the single-source "wrong" rule ------------------------

WIKI = "https://en.wikipedia.org/wiki/Liverpool_2019-20"
ESPN = "https://www.espn.com/soccer/story/liverpool-title"


async def test_one_confirmed_part_does_not_verify_the_whole_claim():
    llm = FakeLLM(
        {
            "factcheck.queries": {"queries": ["liverpool 99 points"]},
            "factcheck.assess": assess(src(BBC, "supports", QUOTE)),
            # The judge says "verified" overall but admits one part is not.
            "factcheck.judge": {
                **judge("verified", 0.9),
                "parts": [
                    {"part": "Liverpool won the 2019-20 title", "verdict": "verified"},
                    {"part": "with 99 points", "verdict": "unsupported"},
                ],
            },
        }
    )
    ctx = _ctx(
        llm, FakeSearch(default=[hit(BBC)]), FakeFetcher({BBC: BBC_TEXT}), (POINTS, 0.6, "c1")
    )
    result = await FactCheckAgent().run(ctx)
    verdict = verdicts(result)["c1"]
    assert verdict.verdict is Verdict.UNSUPPORTED
    assert verdict.note.startswith("Not confirmed: with 99 points.")
    [finding] = result.findings
    assert finding.severity is Severity.CONSIDER  # backed in part, disputed nowhere


async def test_a_contradicted_part_makes_the_claim_wrong():
    llm = FakeLLM(
        {
            "factcheck.queries": {"queries": ["liverpool 99 points"]},
            "factcheck.assess": assess(
                src(BBC, "contradicts", QUOTE), src(WIKI, "contradicts", QUOTE)
            ),
            "factcheck.judge": {
                **judge("verified", 0.9, correction="It was 97 points."),
                "parts": [
                    {"part": "won the title", "verdict": "verified"},
                    {"part": "99 points", "verdict": "wrong"},
                ],
            },
        }
    )
    ctx = _ctx(
        llm,
        FakeSearch(default=[hit(BBC, rank=1), hit(WIKI, rank=2)]),
        FakeFetcher({BBC: BBC_TEXT, WIKI: BBC_TEXT}),
        (POINTS, 0.6, "c1"),
    )
    result = await FactCheckAgent().run(ctx)
    assert verdicts(result)["c1"].verdict is Verdict.WRONG
    assert result.findings[0].severity is Severity.FACTUAL_ERROR


async def test_research_continues_while_parts_are_unconfirmed():
    llm = FakeLLM(
        {
            "factcheck.queries": {"queries": ["liverpool title 2019-20"]},
            "factcheck.assess": [
                # Two independent sources support the claim, but one part is open.
                assess(
                    src(BBC, "supports", QUOTE),
                    src(WIKI, "supports", QUOTE),
                    next_step="refine",
                    next_query="liverpool 99 points total",
                    unverified_parts=["99 points"],
                ),
                assess(src(ESPN, "supports", QUOTE)),
            ],
            "factcheck.judge": judge("verified", 0.9),
        }
    )
    search = FakeSearch(
        {
            "liverpool title 2019-20": [hit(BBC, rank=1), hit(WIKI, rank=2)],
            "99 points total": [hit(ESPN)],
        }
    )
    fetcher = FakeFetcher({BBC: BBC_TEXT, WIKI: BBC_TEXT, ESPN: BBC_TEXT})
    ctx = _ctx(llm, search, fetcher, (POINTS, 0.6, "c1"))
    result = await FactCheckAgent().run(ctx)
    # Without open parts the loop would have stopped after the first search.
    assert search.queries == ["liverpool title 2019-20", "liverpool 99 points total"]
    assert verdicts(result)["c1"].verdict is Verdict.VERIFIED
    second = llm.calls_for("factcheck.assess")[1].messages[-1].content
    assert "Parts still unconfirmed before these sources: 99 points" in second


async def _single_contradiction(url: str, contested: bool) -> Any:
    llm = FakeLLM(
        {
            "factcheck.queries": {"queries": ["liverpool 99 points"]},
            "factcheck.assess": assess(src(url, "contradicts", QUOTE)),
            "factcheck.judge": {
                **judge("wrong", 0.85, correction="It was 97 points."),
                "contested": contested,
            },
        }
    )
    ctx = _ctx(
        llm, FakeSearch(default=[hit(url)]), FakeFetcher({url: BBC_TEXT}), (POINTS, 0.6, "c1")
    )
    return await FactCheckAgent().run(ctx)


async def test_one_reputable_uncontested_contradiction_is_wrong():
    result = await _single_contradiction(BBC, contested=False)
    verdict = verdicts(result)["c1"]
    assert verdict.verdict is Verdict.WRONG
    assert verdict.confidence == 0.7
    assert "One reputable source contradicts this" in verdict.note
    [finding] = result.findings
    assert finding.severity is Severity.FACTUAL_ERROR
    assert finding.suggestion == "It was 97 points."


async def test_contested_or_unknown_source_contradiction_is_not_wrong():
    for url, contested in ((BBC, True), (BLOG, False)):
        result = await _single_contradiction(url, contested)
        assert verdicts(result)["c1"].verdict is Verdict.UNSUPPORTED
        assert result.findings[0].severity is Severity.UNSUPPORTED  # still a must-fix
