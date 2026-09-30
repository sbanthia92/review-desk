"""``reviewdesk review <file-or-url>``: run a full review from the terminal.

Progress goes to stderr, one line per event; the markdown report (or, with
``--json``, the ``Report`` JSON) goes to stdout, so ``> report.md`` works.

Keys come from the environment, optionally loaded from a ``.env`` file
(``--env-file``, default ``./.env`` when it exists; variables already set in
the process win). See ``reviewdesk.registry`` for the variables.

``run(argv, registry_factory=...)`` is the testable entry point: tests inject
a factory that builds a registry of fakes.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from reviewdesk.contracts import (
    Document,
    DocumentTooLong,
    Profile,
    ProgressEvent,
    Report,
)
from reviewdesk.orchestrator import AgentRegistry

from .envfile import load_env_file, merge_env
from .errors import (
    EXIT_FAILED,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_USAGE,
    InputError,
    describe,
    too_long_message,
)

DEFAULT_MAX_WORDS = 20_000
"""Same cap as the local MCP server (``reviewdesk.mcp_local.server``)."""
MAX_WORDS_ENV = "REVIEWDESK_MAX_WORDS"
DEFAULT_ENV_FILE = ".env"
MAX_FOCUS_CHARS = 1_000
MAX_URL_CHARS = 2_048
MAX_STYLE_GUIDE_CHARS = 20_000
PROFILE_CHOICES = [p.value for p in Profile]

RegistryFactory = Callable[[Mapping[str, str]], Any]
"""``env -> AgentRegistry`` (or an awaitable of one); may raise ``MissingSetup``."""


def _real_factory(env: Mapping[str, str]) -> AgentRegistry:
    from reviewdesk.registry import build_registry_from_env

    return build_registry_from_env(env)


def _fake_factory(env: Mapping[str, str]) -> AgentRegistry:
    from reviewdesk.mcp_local.demo import demo_registry

    return demo_registry(delay=0.0)


class _Parser(argparse.ArgumentParser):
    """``ArgumentParser`` whose usage errors return instead of exiting."""

    def error(self, message: str) -> Any:
        raise _UsageError(message)


class _UsageError(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    """The ``reviewdesk`` argument parser."""
    parser = _Parser(
        prog="reviewdesk",
        description="Evidence-backed, multi-agent document review.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command", parser_class=_Parser)
    review = sub.add_parser(
        "review",
        help="review a local markdown/text file or a public http(s) URL",
        description=(
            "Review a document. Progress goes to stderr; the markdown report goes to stdout."
        ),
    )
    review.add_argument("source", help="path to a markdown/text file, or an http(s) URL")
    review.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default=Profile.AUTO.value,
        help="review profile (default: auto-detect)",
    )
    review.add_argument("--focus", help='free-text steer, e.g. "check the financial figures"')
    review.add_argument("--style-guide", metavar="FILE", help="style guide for the copy editor")
    review.add_argument(
        "--max-words",
        type=int,
        metavar="N",
        help=f"word cap (default: ${MAX_WORDS_ENV} or {DEFAULT_MAX_WORDS:,})",
    )
    review.add_argument(
        "--json", action="store_true", help="print the report as JSON instead of markdown"
    )
    review.add_argument("--quiet", action="store_true", help="do not print progress")
    review.add_argument(
        "--env-file",
        metavar="FILE",
        help=f"load variables from FILE (default: ./{DEFAULT_ENV_FILE} if present)",
    )
    review.add_argument(
        "--no-env-file", action="store_true", help=f"do not load ./{DEFAULT_ENV_FILE}"
    )
    # Hidden: run on the offline demo registry (no keys, no network).
    review.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    return parser


def _is_url(source: str) -> bool:
    return source.lower().startswith(("http://", "https://"))


def _read_text(path_arg: str, what: str) -> str:
    path = Path(path_arg).expanduser()
    if not path.exists():
        raise InputError(f"{what} not found: {path_arg}")
    if not path.is_file():
        raise InputError(f"{what} is not a file: {path_arg}")
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise InputError(f"{what} is not UTF-8 text: {path_arg}") from None
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise InputError(f"could not read {what} {path_arg}: {reason}") from None


def _max_words(arg: int | None, env: Mapping[str, str]) -> int:
    if arg is not None:
        if arg < 1:
            raise InputError("--max-words must be a positive integer")
        return arg
    raw = (env.get(MAX_WORDS_ENV) or "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return DEFAULT_MAX_WORDS
        if value > 0:
            return value
    return DEFAULT_MAX_WORDS


def _check_size(doc: Document, limit: int) -> None:
    if doc.word_count > limit:
        raise DocumentTooLong(too_long_message(doc.word_count, limit))


def _load_env(args: argparse.Namespace, environ: Mapping[str, str]) -> dict[str, str]:
    if args.env_file:
        path = Path(args.env_file).expanduser()
        if not path.is_file():
            raise InputError(f"env file not found: {args.env_file}")
    elif args.no_env_file:
        return dict(environ)
    else:
        path = Path(DEFAULT_ENV_FILE)
        if not path.is_file():
            return dict(environ)
    try:
        values = load_env_file(path)
    except (OSError, UnicodeDecodeError):
        raise InputError(f"could not read env file {path}") from None
    return merge_env(values, environ)


class ProgressPrinter:
    """``on_progress`` that writes one line per event to a stream."""

    def __init__(self, stream: TextIO, *, quiet: bool = False) -> None:
        self._stream = stream
        self._quiet = quiet

    def __call__(self, event: ProgressEvent) -> None:
        self.emit(event.step, event.percent, event.message)

    def emit(self, step: str, percent: float, message: str) -> None:
        if self._quiet:
            return
        line = f"[{percent:3.0f}%] {' '.join(step.split())}: {' '.join(message.split())}"
        try:
            self._stream.write(line + "\n")
            self._stream.flush()
        except (OSError, ValueError):
            pass  # progress must never break a review


def render_report(report: Report, doc: Document, *, as_json: bool) -> str:
    """The stdout text for ``report``."""
    if as_json:
        return report.model_dump_json(indent=2)
    from reviewdesk.report import render_markdown

    return render_markdown(report, doc if report.document_id == doc.id else None)


async def _review(
    args: argparse.Namespace,
    env: Mapping[str, str],
    factory: RegistryFactory,
    stdout: TextIO,
    progress: ProgressPrinter,
) -> None:
    from reviewdesk.pipeline import run_review
    from reviewdesk.registry import aclose_registry

    limit = _max_words(args.max_words, env)
    focus = " ".join(args.focus.split()) if args.focus else None
    if focus and len(focus) > MAX_FOCUS_CHARS:
        raise InputError(f"--focus is longer than {MAX_FOCUS_CHARS} characters")
    style_guide: str | None = None
    if args.style_guide:
        style_guide = _read_text(args.style_guide, "style guide").strip() or None
        if style_guide and len(style_guide) > MAX_STYLE_GUIDE_CHARS:
            raise InputError(f"style guide is longer than {MAX_STYLE_GUIDE_CHARS:,} characters")

    source: str = args.source
    doc: Document | None = None
    if _is_url(source):
        if len(source) > MAX_URL_CHARS:
            raise InputError(f"URL is longer than {MAX_URL_CHARS} characters")
    else:
        if "://" in source:
            raise InputError("only http(s) URLs can be fetched")
        text = _read_text(source, "file")
        if not text.strip():
            raise InputError(f"file is empty: {source}")
        doc = Document.from_text(text)
        _check_size(doc, limit)

    built = factory(env)
    if inspect.isawaitable(built):
        built = await built
    if not isinstance(built, AgentRegistry):
        raise TypeError("registry factory did not return an AgentRegistry")
    registry: AgentRegistry = built
    try:
        if doc is None:
            progress.emit("fetching", 0.0, "Fetching the URL")
            page = await registry.fetcher.fetch(source)
            if not page.text.strip():
                raise InputError("the page has no readable text")
            doc = Document.from_text(page.text, source_url=page.final_url or source)
            _check_size(doc, limit)
        report = await run_review(
            doc,
            Profile(args.profile),
            progress,
            registry=registry,
            focus=focus,
            style_guide=style_guide,
        )
    finally:
        await aclose_registry(registry)
    output = render_report(report, doc, as_json=args.json)
    stdout.write(output if output.endswith("\n") else output + "\n")
    stdout.flush()


def run(
    argv: Sequence[str] | None = None,
    *,
    registry_factory: RegistryFactory | None = None,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the CLI and return the exit code (never raises for expected errors)."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except _UsageError as exc:
        err.write(parser.format_usage())
        err.write(f"reviewdesk: error: {exc}\n")
        return EXIT_USAGE
    except SystemExit as exc:  # --help
        code = exc.code
        return code if isinstance(code, int) else EXIT_OK
    if args.command != "review":
        err.write(parser.format_help())
        return EXIT_USAGE

    url = args.source if _is_url(args.source) else None
    try:
        env = _load_env(args, os.environ if environ is None else environ)
        if registry_factory is not None:
            factory = registry_factory
        elif args.fake:
            factory = _fake_factory
        else:
            factory = _real_factory
        progress = ProgressPrinter(err, quiet=args.quiet)
        asyncio.run(_review(args, env, factory, out, progress))
    except KeyboardInterrupt:
        err.write("reviewdesk: interrupted\n")
        return EXIT_INTERRUPTED
    except Exception as exc:
        message, code = describe(exc, url=url)
        err.write(f"reviewdesk: {message}\n")
        return code
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point (``reviewdesk``)."""
    code = run(argv)
    raise SystemExit(code if code is not None else EXIT_FAILED)
