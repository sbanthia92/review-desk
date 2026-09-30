"""Compare search providers on claim-style queries for the Phase 0 decision.

Runs a fixed list of queries against every provider that has a key in the
environment (``BRAVE_SEARCH_API_KEY``, ``TAVILY_API_KEY``, ``EXA_API_KEY``;
providers without a key are skipped) and writes a markdown table: one row
per query x provider with the top results, latency, and an empty
"relevance (1–5, by hand)" column to fill in.

Usage::

    uv run python -m scripts.compare_search [OUTPUT] [--k 5] [--providers brave,exa]
        [--env-file .env]

``OUTPUT`` defaults to ``eval/results/search_comparison-<UTC timestamp>.md``
(``eval/results/`` is gitignored). Before deciding, re-confirm each
provider's free-tier limits on its pricing page and note them in the file.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from reviewdesk.contracts import ReviewDeskError, SearchClient, SearchResult
from reviewdesk.providers.search.config import PROVIDERS, search_clients_from_env

QUERIES: tuple[str, ...] = (
    # Football / blog-style claims.
    "Arsenal went the entire 2003-04 Premier League season unbeaten",
    "Erling Haaland scored 36 Premier League goals in the 2022-23 season",
    "Leicester City were 5000-1 outsiders to win the 2015-16 Premier League",
    "Pep Guardiola's Barcelona won six trophies in 2009",
    "high pressing teams concede fewer goals from open play",
    # Technical / design-doc claims.
    "PostgreSQL SERIALIZABLE isolation uses serializable snapshot isolation",
    "HTTP/3 runs over QUIC instead of TCP",
    "Python 3.12 removed the distutils module from the standard library",
    "SQLite supports only one writer at a time",
    "envelope encryption wraps a data key with a KMS master key",
)
"""Fixed claim-style queries: half football/blog-style, half technical."""

DEFAULT_K = 5
DEFAULT_RESULTS_DIR = Path("eval/results")


@dataclass
class Cell:
    """The outcome of one query against one provider."""

    query: str
    provider: str
    results: list[SearchResult] = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None


async def run_query(name: str, client: SearchClient, query: str, k: int) -> Cell:
    """Run one query, capturing results, latency and any error message."""
    started = time.perf_counter()
    try:
        results = await client.search(query, k=k)
    except ReviewDeskError as exc:
        return Cell(query, name, seconds=time.perf_counter() - started, error=str(exc))
    return Cell(query, name, results=list(results), seconds=time.perf_counter() - started)


async def run_comparison(
    clients: Mapping[str, SearchClient],
    queries: Sequence[str] = QUERIES,
    *,
    k: int = DEFAULT_K,
) -> list[Cell]:
    """Run every query against every client, one provider call at a time.

    Calls are sequential so free-tier per-second rate limits are not tripped
    and latencies are comparable.
    """
    cells: list[Cell] = []
    for query in queries:
        for name, client in clients.items():
            cells.append(await run_query(name, client, query, k))
    return cells


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


def _link(hit: SearchResult) -> str:
    title = _escape(hit.title) or "(untitled)"
    url = hit.url.replace(" ", "%20").replace("|", "%7C").replace(")", "%29")
    return f"{hit.rank}. [{title}]({url})"


def render_markdown(
    cells: Sequence[Cell],
    providers: Sequence[str],
    *,
    k: int = DEFAULT_K,
    generated_at: datetime | None = None,
) -> str:
    """Render the comparison as markdown (summary, then the per-query table)."""
    stamp = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Search provider comparison",
        "",
        f"Generated {stamp}. Providers: {', '.join(providers) or 'none'}. Top {k} results.",
        "",
        "Fill in the relevance column by hand (1 = useless, 5 = primary source that",
        "settles the claim). Re-confirm each provider's free-tier limits before deciding.",
        "",
        "## Summary",
        "",
        "| Provider | Queries | Errors | Mean results | Mean latency (s) | Free tier (confirm) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for provider in providers:
        mine = [c for c in cells if c.provider == provider]
        errors = sum(1 for c in mine if c.error)
        ok = [c for c in mine if not c.error]
        mean_results = sum(len(c.results) for c in ok) / len(ok) if ok else 0.0
        mean_latency = sum(c.seconds for c in mine) / len(mine) if mine else 0.0
        lines.append(
            f"| {provider} | {len(mine)} | {errors} | {mean_results:.1f} | {mean_latency:.2f} | |"
        )
    lines += [
        "",
        "## Results",
        "",
        "| Query | Provider | Top results | Latency (s) | Relevance (1–5, by hand) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cell in cells:
        if cell.error:
            top = f"ERROR: {_escape(cell.error)}"
        elif not cell.results:
            top = "(no results)"
        else:
            top = "<br>".join(_link(hit) for hit in cell.results)
        lines.append(f"| {_escape(cell.query)} | {cell.provider} | {top} | {cell.seconds:.2f} | |")
    lines.append("")
    return "\n".join(lines)


def read_env_file(path: Path) -> dict[str, str]:
    """Parse simple ``KEY=value`` lines (comments and blanks ignored)."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        values[key] = value.strip().strip("'\"")
    return values


def default_output(now: datetime | None = None) -> Path:
    """``eval/results/search_comparison-<timestamp>.md``."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RESULTS_DIR / f"search_comparison-{stamp}.md"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("output", nargs="?", type=Path, help="markdown file to write")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="results per query")
    parser.add_argument("--providers", help=f"comma-separated subset of: {', '.join(PROVIDERS)}")
    parser.add_argument("--env-file", type=Path, help="read keys from this file too")
    return parser.parse_args(argv)


async def amain(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    clients: Mapping[str, SearchClient] | None = None,
) -> int:
    """Entry point. ``env`` and ``clients`` are injectable for tests."""
    args = _parse_args(argv)
    if args.k < 1:
        print("--k must be at least 1", file=sys.stderr)
        return 2
    owned: list[object] = []
    if clients is None:
        source: dict[str, str] = dict(os.environ if env is None else env)
        if args.env_file is not None:
            source = {**read_env_file(args.env_file), **source}
        only = [p.strip() for p in args.providers.split(",")] if args.providers else None
        try:
            built = search_clients_from_env(source, only=only)
        except ReviewDeskError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        owned.extend(built.values())
        clients = dict(built)
        skipped = [name for name in (only or PROVIDERS) if name not in clients]
        for name in skipped:
            print(f"skipping {name}: {PROVIDERS[name][0]} is not set", file=sys.stderr)
    if not clients:
        print("no search providers configured; set at least one API key", file=sys.stderr)
        return 2

    output: Path = args.output or default_output()
    try:
        cells = await run_comparison(clients, QUERIES, k=args.k)
    finally:
        for client in owned:
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()
    await asyncio.to_thread(_write, output, render_markdown(cells, list(clients), k=args.k))
    print(f"wrote {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous wrapper for ``amain``."""
    return asyncio.run(amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
