"""Run ``ReviewPipeline``s over the seeded dataset and score every report.

``run_eval`` accepts any callables of the ``run_review`` shape
(``(doc, profile, on_progress) -> Awaitable[Report]``), keyed by a display
name (e.g. ``baseline``, ``fixed``, ``full``). Every document runs
``repeats`` times per pipeline. A pipeline that raises or times out is scored
as an empty report with ``error`` set, so failures lower recall instead of
aborting the run. The exception is a fatal provider error (rejected key, no
credit, missing setup): every later call would fail and cost time, so the run
stops, keeps the scores it has and records why in ``EvalResults.aborted``.
Each score is also appended to a checkpoint file as it completes, so a run
that dies still leaves its results on disk. Wall-clock seconds are measured
here, around the pipeline call only (scoring and judging are excluded).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from eval.dataset import SeededDocument
from eval.judge import EvalJudge
from eval.metrics import DocScore, score_report
from eval.results import EvalResults
from reviewdesk.contracts import Profile, ProgressEvent, Report, ReviewPipeline

RESULTS_DIR = Path(__file__).parent / "results"
"""Where run outputs are written (gitignored)."""

FATAL_ERRORS = ("AuthError", "MissingSetup")
"""Error types that stop the whole run: retrying cannot help."""


def is_fatal(error: str | None) -> bool:
    """True if a run's error means every further run would fail the same way."""
    return error is not None and error.startswith(FATAL_ERRORS)


def _ignore_progress(event: ProgressEvent) -> None:
    return None


def _error_text(exc: BaseException) -> str:
    """Type and a short message; never more than 200 characters."""
    message = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {message}"[:200] if message else type(exc).__name__


async def run_one(
    name: str,
    pipeline: ReviewPipeline,
    seeded: SeededDocument,
    *,
    repeat: int = 0,
    judge: EvalJudge | None = None,
    time_limit: float | None = None,
    auto_profile: bool = False,
    reports_dir: Path | None = None,
) -> DocScore:
    """Run one pipeline once on one document and score the report.

    With ``reports_dir``, the report itself is saved there as
    ``<pipeline>-<doc id>-r<repeat>.json`` so findings can be inspected later.
    """
    doc = seeded.document
    profile = Profile.AUTO if auto_profile else seeded.profile
    error: str | None = None
    started = time.monotonic()
    try:
        report = await asyncio.wait_for(pipeline(doc, profile, _ignore_progress), time_limit)
    except Exception as exc:  # scored as an empty report
        error = "timeout" if isinstance(exc, TimeoutError) else _error_text(exc)
        report = Report(document_id=doc.id, profile=seeded.profile, verdict_line="(failed)")
    seconds = time.monotonic() - started
    if reports_dir is not None and error is None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"{name}-{seeded.id}-r{repeat}.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return await score_report(
        seeded, report, pipeline=name, repeat=repeat, seconds=seconds, error=error, judge=judge
    )


async def run_eval(
    pipelines: Mapping[str, ReviewPipeline],
    dataset: Sequence[SeededDocument],
    *,
    repeats: int = 1,
    judge: EvalJudge | None = None,
    concurrency: int = 1,
    time_limit: float | None = None,
    auto_profile: bool = False,
    on_score: Callable[[DocScore], None] | None = None,
    checkpoint: Path | None = None,
    reports_dir: Path | None = None,
) -> EvalResults:
    """Run every pipeline ``repeats`` times over ``dataset`` and collect scores.

    ``concurrency`` bounds how many (pipeline, document, repeat) runs are in
    flight at once. ``auto_profile`` passes ``Profile.AUTO`` instead of the
    answer key's profile. ``on_score`` is called as each score completes.
    ``checkpoint`` is a JSON-lines file each score is appended to as it
    completes. A fatal provider error stops the run early (see module docstring).
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if not pipelines:
        raise ValueError("no pipelines to run")
    gate = asyncio.Semaphore(max(1, concurrency))
    aborted: list[str] = []
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    async def task(
        name: str, pipeline: ReviewPipeline, seeded: SeededDocument, r: int
    ) -> DocScore | None:
        async with gate:
            if aborted:
                return None  # the run was stopped; do not start more work
            score = await run_one(
                name,
                pipeline,
                seeded,
                repeat=r,
                judge=judge,
                time_limit=time_limit,
                auto_profile=auto_profile,
                reports_dir=reports_dir,
            )
        if is_fatal(score.error):
            aborted.append(score.error or "fatal error")
        if checkpoint is not None:
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(score.model_dump_json() + "\n")
        if on_score is not None:
            on_score(score)
        return score

    jobs = [
        task(name, pipeline, seeded, r)
        for name, pipeline in pipelines.items()
        for r in range(repeats)
        for seeded in dataset
    ]
    scores = await asyncio.gather(*jobs)
    return EvalResults(
        pipelines=list(pipelines),
        doc_ids=[d.id for d in dataset],
        repeats=repeats,
        judged=judge is not None,
        # A fatal error is the reason the run stopped, not a result worth scoring.
        scores=[s for s in scores if s is not None and not is_fatal(s.error)],
        aborted=aborted[0] if aborted else None,
    )


def save_results(results: EvalResults, directory: Path | None = None) -> tuple[Path, Path]:
    """Write ``<stamp>.json`` (all scores) and ``<stamp>.md`` (the table)."""
    root = directory or RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    stamp = results.started_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = root / f"{stamp}.json"
    md_path = root / f"{stamp}.md"
    json_path.write_text(results.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(results.table() + "\n", encoding="utf-8")
    return json_path, md_path
