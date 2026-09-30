"""Command line: ``uv run python -m eval <command>``.

Commands:

- ``validate``: load the dataset and check every answer-key span.
- ``list``: list the seeded documents and their defect counts.
- ``run``: run pipelines over the dataset and print the results table.

Pipelines (``-p``, repeatable) are either built-in names or import specs:

- ``oracle``, ``oracle-no-rules``, ``noisy``, ``empty``: offline
  ``FakeAgent`` pipelines (see ``eval.fake_pipeline``).
- ``baseline``: the single-prompt reviewer; needs ``--llm``.
- ``NAME=module:attr`` or ``module:attr``: any ``ReviewPipeline`` callable,
  e.g. ``full=reviewdesk.pipeline:run_review``.

``--llm module:attr`` names a zero-argument callable returning an
``LLMClient`` (reads its own key from the environment). It powers the baseline
and, with ``--judge``, the LLM judge. Keys are never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from eval.baseline import BaselineReviewer
from eval.dataset import DatasetError, DefectType, SeededDocument, load_dataset
from eval.fake_pipeline import builtin_pipelines
from eval.harness import RESULTS_DIR, run_eval, save_results
from eval.judge import LLMJudge
from eval.metrics import DocScore
from reviewdesk.contracts import LLMClient, ReviewPipeline


class CLIError(Exception):
    """A usage error shown to the user without a traceback."""


def import_spec(spec: str) -> Any:
    """Import ``module:attr`` (attr may be dotted)."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise CLIError(f"expected module:attr, got {spec!r}")
    try:
        obj: Any = importlib.import_module(module_name)
        for part in attr.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as exc:
        raise CLIError(f"cannot import {spec!r}: {exc}") from exc
    return obj


def make_llm(spec: str | None) -> LLMClient | None:
    """Build the LLM client from ``--llm``, or None."""
    if spec is None:
        return None
    factory = import_spec(spec)
    if not callable(factory):
        raise CLIError(f"{spec!r} is not callable")
    return cast(LLMClient, factory())


def resolve_pipelines(
    specs: Sequence[str], dataset: Sequence[SeededDocument], llm: LLMClient | None
) -> dict[str, ReviewPipeline]:
    """Map ``-p`` values to named pipelines."""
    builtins = builtin_pipelines(dataset)
    out: dict[str, ReviewPipeline] = {}
    for spec in specs:
        name, sep, target = spec.partition("=")
        if not sep:
            name, target = spec, spec
        if target in builtins:
            out[name] = builtins[target]
        elif target == "baseline":
            if llm is None:
                raise CLIError("the baseline pipeline needs --llm module:factory")
            out[name] = BaselineReviewer(llm)
        elif ":" in target:
            obj = import_spec(target)
            if not callable(obj):
                raise CLIError(f"{target!r} is not callable")
            out[name if sep else target.rpartition(":")[2]] = cast(ReviewPipeline, obj)
        else:
            known = ", ".join([*builtins, "baseline"])
            raise CLIError(f"unknown pipeline {target!r}; use one of {known} or module:attr")
    return out


def _progress(score: DocScore) -> None:
    status = f"error: {score.error}" if score.error else "ok"
    caught = sum(o.caught for o in score.defects)
    print(
        f"  {score.pipeline} r{score.repeat} {score.doc_id}: "
        f"{caught}/{len(score.defects)} caught, {score.seconds:.1f}s, {status}",
        file=sys.stderr,
    )


def cmd_validate(args: argparse.Namespace) -> int:
    docs = load_dataset(args.data)
    total = sum(len(d.defects) for d in docs)
    print(f"ok: {len(docs)} documents, {total} seeded defects, all spans valid")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    docs = load_dataset(args.data)
    types = list(DefectType)
    print("| id | profile | words | " + " | ".join(t.value for t in types) + " |")
    print("|" + "---|" * (3 + len(types)))
    for d in docs:
        counts = [str(len(d.of_type(t))) for t in types]
        row = [d.id, d.profile.value, str(d.document.word_count), *counts]
        print("| " + " | ".join(row) + " |")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    ids = [i for i in args.docs.split(",") if i] if args.docs else None
    dataset = load_dataset(args.data, ids=ids)
    llm = make_llm(args.llm)
    if args.judge and llm is None:
        raise CLIError("--judge needs --llm module:factory")
    pipelines = resolve_pipelines(args.pipeline or ["oracle"], dataset, llm)
    judge = LLMJudge(llm) if args.judge and llm is not None else None
    results = asyncio.run(
        run_eval(
            pipelines,
            dataset,
            repeats=args.repeats,
            judge=judge,
            concurrency=args.concurrency,
            time_limit=args.timeout,
            auto_profile=args.auto_profile,
            on_score=None if args.quiet else _progress,
        )
    )
    print(results.table())
    if not args.no_save:
        json_path, md_path = save_results(results, args.out)
        print(f"\nwrote {json_path} and {md_path}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval", description="Review Desk eval")
    parser.add_argument("--data", type=Path, default=None, help="dataset directory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="validate the dataset").set_defaults(func=cmd_validate)
    sub.add_parser("list", help="list seeded documents").set_defaults(func=cmd_list)
    run = sub.add_parser("run", help="run pipelines and print the results table")
    run.add_argument("-p", "--pipeline", action="append", help="pipeline name or NAME=mod:attr")
    run.add_argument("--repeats", type=int, default=1, help="runs per document (default 1)")
    run.add_argument("--docs", default="", help="comma-separated document ids")
    run.add_argument("--llm", default=None, help="module:factory returning an LLMClient")
    run.add_argument("--judge", action="store_true", help="use the LLM judge (needs --llm)")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--timeout", type=float, default=None, help="seconds per document run")
    run.add_argument("--auto-profile", action="store_true", help="pass profile=auto")
    run.add_argument("--out", type=Path, default=RESULTS_DIR, help="results directory")
    run.add_argument("--no-save", action="store_true", help="do not write result files")
    run.add_argument("-q", "--quiet", action="store_true", help="no per-document progress")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    func = cast(Callable[[argparse.Namespace], int], args.func)
    try:
        return func(args)
    except (CLIError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
