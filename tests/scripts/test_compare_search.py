from datetime import UTC, datetime

import httpx

from reviewdesk.contracts import RateLimitError, SearchResult
from reviewdesk.providers.search.config import search_clients_from_env
from reviewdesk.testing.fakes import FakeSearch
from scripts import compare_search
from scripts.compare_search import (
    QUERIES,
    Cell,
    amain,
    default_output,
    read_env_file,
    render_markdown,
    run_comparison,
)


def hits(prefix: str, n: int = 3) -> list[SearchResult]:
    return [
        SearchResult(url=f"https://{prefix}.example/{i}", title=f"{prefix} | result {i}", rank=i)
        for i in range(1, n + 1)
    ]


def test_fixed_query_set():
    assert 8 <= len(QUERIES) <= 12
    assert len(set(QUERIES)) == len(QUERIES)
    assert any("Arsenal" in q for q in QUERIES)
    assert any("QUIC" in q for q in QUERIES)


async def test_run_comparison_covers_every_query_and_provider():
    clients = {"alpha": FakeSearch(default=hits("a")), "beta": FakeSearch(default=hits("b", 1))}
    cells = await run_comparison(clients, QUERIES, k=2)
    assert len(cells) == len(QUERIES) * 2
    assert [c.provider for c in cells[:2]] == ["alpha", "beta"]
    assert all(len(c.results) == 2 for c in cells if c.provider == "alpha")
    assert clients["alpha"].queries == list(QUERIES)


async def test_errors_are_captured_per_cell():
    clients = {"bad": FakeSearch(error=RateLimitError("bad: rate limited (HTTP 429)"))}
    cells = await run_comparison(clients, ["q1"], k=3)
    assert cells[0].error == "bad: rate limited (HTTP 429)"
    assert cells[0].results == []


def test_render_markdown_table():
    cells = [
        Cell("claim | one", "alpha", results=hits("a", 2), seconds=0.1234),
        Cell("claim | one", "beta", error="beta: server error (HTTP 500)", seconds=1.0),
        Cell("claim two", "alpha", seconds=0.2),
        Cell("claim two", "beta", results=hits("b", 1), seconds=0.3),
    ]
    md = render_markdown(
        cells, ["alpha", "beta"], k=2, generated_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    )
    assert "Generated 2026-01-02 03:04 UTC" in md
    assert "| Query | Provider | Top results | Latency (s) | Relevance (1–5, by hand) |" in md
    assert "| alpha | 2 | 0 | 1.0 | 0.16 | |" in md
    assert "| beta | 2 | 1 | 1.0 | 0.65 | |" in md
    assert (
        "| claim \\| one | alpha | 1. [a \\| result 1](https://a.example/1)<br>"
        "2. [a \\| result 2](https://a.example/2) | 0.12 | |"
    ) in md
    assert "| claim \\| one | beta | ERROR: beta: server error (HTTP 500) | 1.00 | |" in md
    assert "| claim two | alpha | (no results) | 0.20 | |" in md
    # Every table row has the same number of unescaped pipes as the header.
    table = [line for line in md.splitlines() if line.startswith("| claim")]
    assert all(line.replace("\\|", "").count("|") == 6 for line in table)


async def test_amain_with_injected_clients_writes_table(tmp_path, capsys):
    out = tmp_path / "nested" / "cmp.md"
    clients = {"alpha": FakeSearch(default=hits("a")), "beta": FakeSearch(default=hits("b"))}
    code = await amain([str(out), "--k", "2"], clients=clients)
    assert code == 0
    md = out.read_text()
    assert md.count("| alpha |") == len(QUERIES) + 1  # rows plus summary
    assert md.count("| beta |") == len(QUERIES) + 1
    assert "wrote" in capsys.readouterr().out


async def test_amain_builds_clients_from_env_and_skips_missing(tmp_path, capsys, monkeypatch):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"results": [{"title": "T", "url": "https://t.example/", "content": "c"}]}
        )

    def with_mock(env, *, only=None):
        return search_clients_from_env(env, only=only, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(compare_search, "search_clients_from_env", with_mock)
    out = tmp_path / "cmp.md"
    code = await amain([str(out)], env={"TAVILY_API_KEY": "tv-SECRET"})
    assert code == 0
    assert len(requests) == len(QUERIES)
    err = capsys.readouterr().err
    assert "skipping brave: BRAVE_SEARCH_API_KEY is not set" in err
    assert "skipping exa" in err
    md = out.read_text()
    assert "[T](https://t.example/)" in md
    assert "tv-SECRET" not in md
    assert "tv-SECRET" not in err


async def test_amain_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nexport EXA_API_KEY='exa-key'\nOTHER=1\n")
    seen: dict[str, str] = {}

    def capture(env, *, only=None):
        seen.update(env)
        return {}

    monkeypatch.setattr(compare_search, "search_clients_from_env", capture)
    code = await amain([str(tmp_path / "o.md"), "--env-file", str(env_file)], env={})
    assert code == 2  # nothing built by the stub
    assert seen["EXA_API_KEY"] == "exa-key"


async def test_amain_without_keys_fails_cleanly(tmp_path, capsys):
    code = await amain([str(tmp_path / "o.md")], env={})
    assert code == 2
    assert "no search providers configured" in capsys.readouterr().err
    assert not (tmp_path / "o.md").exists()


async def test_amain_unknown_provider_and_bad_k(tmp_path, capsys):
    assert await amain([str(tmp_path / "o.md"), "--providers", "bing"], env={}) == 2
    assert "unknown search provider" in capsys.readouterr().err
    assert await amain([str(tmp_path / "o.md"), "--k", "0"], env={}) == 2


def test_read_env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text('A=1\n\n# x\nB="two"\nexport C = three\nnot a pair\n')
    assert read_env_file(path) == {"A": "1", "B": "two", "C": "three"}


def test_default_output_under_eval_results():
    path = default_output(datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC))
    assert str(path) == "eval/results/search_comparison-20260930T120000Z.md"
