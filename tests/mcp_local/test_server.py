"""The ``review_document`` tool over an in-process MCP client, on fakes."""

from __future__ import annotations

import pytest
from mcp import Client

from reviewdesk.contracts import (
    AuthError,
    BlockedURLError,
    FetchedPage,
    FetchError,
    MissingSetup,
    Profile,
    ProviderError,
    RateLimitError,
)
from reviewdesk.mcp_local import TOOL_NAME, create_server
from reviewdesk.report import render_markdown
from reviewdesk.testing.fakes import OPINION_DOC, FakeFetcher, empty_report
from tests.mcp_local.helpers import ProgressLog, RecordingPipeline, call, text_of

SECRET_TEXT = "ZEBRA-QUOKKA-7741 confidential paragraph"


# ---------------------------------------------------------------------------
# Tool listing and the happy path
# ---------------------------------------------------------------------------


async def test_lists_exactly_one_tool_with_the_design_inputs() -> None:
    server = create_server(pipeline=RecordingPipeline(), fetcher=FakeFetcher())
    async with Client(server, mode="legacy") as client:
        tools = (await client.list_tools()).tools
    assert [t.name for t in tools] == [TOOL_NAME]
    schema = tools[0].input_schema
    assert set(schema["properties"]) == {"content", "url", "profile", "focus"}
    assert schema.get("required", []) == []
    assert schema["properties"]["profile"]["enum"] == ["auto", "opinion", "design_doc"]
    assert schema["properties"]["profile"]["default"] == "auto"


@pytest.mark.parametrize("mode", ["legacy", "auto"])
async def test_content_returns_markdown_report(mode: str) -> None:
    pipeline = RecordingPipeline()
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher())

    result = await call(server, {"content": OPINION_DOC.text}, mode=mode)

    assert not result.is_error
    [seen] = pipeline.calls
    assert seen.doc.text == OPINION_DOC.text
    assert seen.doc.word_count == OPINION_DOC.word_count
    assert seen.doc.source_url is None
    assert seen.profile is Profile.AUTO
    assert seen.focus is None
    assert text_of(result) == render_markdown(pipeline.returned[0], seen.doc)
    assert text_of(result).startswith("# Review Desk report")


async def test_report_for_another_document_renders_without_quoting_it() -> None:
    pipeline = RecordingPipeline(match_document=False)
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher())
    result = await call(server, {"content": OPINION_DOC.text})
    assert not result.is_error
    assert text_of(result) == render_markdown(pipeline.report)


@pytest.mark.parametrize(
    ("profile", "expected"),
    [("auto", Profile.AUTO), ("opinion", Profile.OPINION), ("design_doc", Profile.DESIGN_DOC)],
)
async def test_profile_and_focus_are_passed_through(profile: str, expected: Profile) -> None:
    pipeline = RecordingPipeline(report=empty_report())
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher())

    result = await call(
        server,
        {"content": OPINION_DOC.text, "profile": profile, "focus": "check the goal figures"},
    )

    assert not result.is_error
    [seen] = pipeline.calls
    assert seen.profile is expected
    assert seen.focus == "check the goal figures"


async def test_blank_focus_becomes_none() -> None:
    pipeline = RecordingPipeline()
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher())
    await call(server, {"content": OPINION_DOC.text, "focus": "   "})
    assert pipeline.calls[0].focus is None


async def test_invalid_profile_is_a_tool_error() -> None:
    pipeline = RecordingPipeline()
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher())
    result = await call(server, {"content": OPINION_DOC.text, "profile": "poem"})
    assert result.is_error
    assert not pipeline.calls


async def test_too_long_focus_is_rejected() -> None:
    pipeline = RecordingPipeline()
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher())
    result = await call(server, {"content": OPINION_DOC.text, "focus": "x" * 1001})
    assert result.is_error
    assert "focus" in text_of(result)
    assert not pipeline.calls


# ---------------------------------------------------------------------------
# Exactly one of content / url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"content": "Some text here.", "url": "https://example.org/post"},
        {"content": "   \n\t "},
        {"url": "  "},
        {"content": "", "url": ""},
    ],
)
async def test_exactly_one_of_content_or_url(arguments: dict[str, str]) -> None:
    pipeline = RecordingPipeline()
    fetcher = FakeFetcher({"https://example.org/post": "A page."})
    server = create_server(pipeline=pipeline, fetcher=fetcher)

    result = await call(server, arguments)

    assert result.is_error
    assert "exactly one of `content`" in text_of(result)
    assert not pipeline.calls
    assert not fetcher.fetched


# ---------------------------------------------------------------------------
# URL input
# ---------------------------------------------------------------------------


async def test_url_is_fetched_and_reviewed() -> None:
    url = "https://example.org/post"
    page = FetchedPage(
        url=url, final_url="https://example.org/post/", status=200, text=OPINION_DOC.text
    )
    fetcher = FakeFetcher({url: page})
    pipeline = RecordingPipeline()
    progress = ProgressLog()
    server = create_server(pipeline=pipeline, fetcher=fetcher)

    result = await call(server, {"url": f"  {url} "}, progress=progress)

    assert not result.is_error
    assert fetcher.fetched == [url]
    [seen] = pipeline.calls
    assert seen.doc.text == OPINION_DOC.text
    assert seen.doc.source_url == "https://example.org/post/"
    assert progress.messages[0] == "Fetching the URL"


async def test_unreachable_url_is_a_clear_error() -> None:
    pipeline = RecordingPipeline()
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher())

    result = await call(server, {"url": "https://nowhere.example/x"})

    assert result.is_error
    message = text_of(result)
    assert "URL unreachable" in message
    assert "https://nowhere.example/x" in message
    assert not pipeline.calls


async def test_blocked_url_is_a_clear_error() -> None:
    url = "http://169.254.169.254/latest/meta-data"
    fetcher = FakeFetcher({url: BlockedURLError("private address")})
    server = create_server(pipeline=RecordingPipeline(), fetcher=fetcher)

    result = await call(server, {"url": url})

    assert result.is_error
    assert "URL unreachable" in text_of(result)
    assert "public http(s)" in text_of(result)


async def test_fetch_error_detail_is_included() -> None:
    url = "https://example.org/gone"
    fetcher = FakeFetcher({url: FetchError("HTTP 404")})
    server = create_server(pipeline=RecordingPipeline(), fetcher=fetcher)
    result = await call(server, {"url": url})
    assert result.is_error
    assert "URL unreachable" in text_of(result)
    assert "HTTP 404" in text_of(result)


async def test_page_without_text_is_an_error() -> None:
    url = "https://example.org/empty"
    server = create_server(pipeline=RecordingPipeline(), fetcher=FakeFetcher({url: "   "}))
    result = await call(server, {"url": url})
    assert result.is_error
    assert "no readable text" in text_of(result)


# ---------------------------------------------------------------------------
# Size cap
# ---------------------------------------------------------------------------


async def test_document_too_long() -> None:
    pipeline = RecordingPipeline()
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher(), max_words=10)

    result = await call(server, {"content": " ".join(["word"] * 11)})

    assert result.is_error
    message = text_of(result)
    assert "Document too long: 11 words; the limit is 10 words" in message
    assert not pipeline.calls


async def test_document_at_the_cap_is_accepted() -> None:
    pipeline = RecordingPipeline()
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher(), max_words=10)
    result = await call(server, {"content": " ".join(["word"] * 10)})
    assert not result.is_error


async def test_fetched_page_is_size_capped_too() -> None:
    url = "https://example.org/long"
    pipeline = RecordingPipeline()
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher({url: "w " * 50}), max_words=20)
    result = await call(server, {"url": url})
    assert result.is_error
    assert "Document too long" in text_of(result)
    assert not pipeline.calls


def test_max_words_must_be_positive() -> None:
    with pytest.raises(ValueError):
        create_server(pipeline=RecordingPipeline(), fetcher=FakeFetcher(), max_words=0)


# ---------------------------------------------------------------------------
# Pipeline errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (MissingSetup("no LLM key"), "Missing setup: set ANTHROPIC_API_KEY (or OPENAI_API_KEY)"),
        (AuthError("anthropic: API key rejected (HTTP 401)"), "Invalid or out-of-credit"),
        (RateLimitError("anthropic: rate limited"), "Rate limit exceeded"),
        (ProviderError("upstream 503", retryable=True), "A provider call failed"),
    ],
)
async def test_pipeline_errors_are_clear_tool_errors(error: Exception, expected: str) -> None:
    server = create_server(pipeline=RecordingPipeline(raises=error), fetcher=FakeFetcher())
    result = await call(server, {"content": OPINION_DOC.text})
    assert result.is_error
    assert expected in text_of(result)


async def test_missing_setup_mentions_search_key_and_fake_mode() -> None:
    server = create_server(
        pipeline=RecordingPipeline(raises=MissingSetup("")), fetcher=FakeFetcher()
    )
    message = text_of(await call(server, {"content": OPINION_DOC.text}))
    assert "search key" in message
    assert "REVIEWDESK_FAKE=1" in message


async def test_unexpected_error_hides_its_message_and_the_document() -> None:
    server = create_server(
        pipeline=RecordingPipeline(raises=RuntimeError(f"boom {SECRET_TEXT}")),
        fetcher=FakeFetcher(),
    )
    result = await call(server, {"content": f"{SECRET_TEXT} and more words here."})
    assert result.is_error
    message = text_of(result)
    assert "RuntimeError" in message
    assert "ZEBRA" not in message


async def test_errors_and_logs_never_contain_document_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("DEBUG")
    content = f"{SECRET_TEXT}. " * 3
    for pipeline, max_words in [
        (RecordingPipeline(raises=AuthError("key rejected")), 1000),
        (RecordingPipeline(), 2),
        (RecordingPipeline(), 1000),
    ]:
        server = create_server(pipeline=pipeline, fetcher=FakeFetcher(), max_words=max_words)
        result = await call(server, {"content": content})
        if result.is_error:
            assert "ZEBRA" not in text_of(result)
    assert "ZEBRA" not in caplog.text


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["legacy", "auto"])
async def test_progress_notifications_arrive_in_order(mode: str) -> None:
    steps = [("Classifying", 0.0), ("Extracting", 10.0), ("Fact-checking", 55.5), ("Done", 100.0)]
    pipeline = RecordingPipeline(progress=steps)
    progress = ProgressLog()
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher())

    result = await call(server, {"content": OPINION_DOC.text}, progress=progress, mode=mode)

    assert not result.is_error
    assert progress.events == [(p, 100.0, m) for m, p in steps]


async def test_progress_is_clamped_and_never_decreases() -> None:
    steps = [("a", 30.0), ("b", 20.0), ("c", 50.0)]
    progress = ProgressLog()
    server = create_server(pipeline=RecordingPipeline(progress=steps), fetcher=FakeFetcher())
    await call(server, {"content": OPINION_DOC.text}, progress=progress)
    assert [p for p, _, _ in progress.events] == [30.0, 30.0, 50.0]


async def test_progress_before_an_error_is_still_delivered() -> None:
    pipeline = RecordingPipeline(progress=[("Extracting", 10.0)], raises=RateLimitError("slow"))
    progress = ProgressLog()
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher())
    result = await call(server, {"content": OPINION_DOC.text}, progress=progress)
    assert result.is_error
    assert progress.messages == ["Extracting"]


async def test_without_a_progress_token_the_review_still_completes() -> None:
    pipeline = RecordingPipeline(progress=[("Extracting", 10.0), ("Done", 100.0)])
    server = create_server(pipeline=pipeline, fetcher=FakeFetcher())
    result = await call(server, {"content": OPINION_DOC.text}, progress=None)
    assert not result.is_error
    assert text_of(result).startswith("# Review Desk report")
