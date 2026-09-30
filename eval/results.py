"""Aggregate ``DocScore``s into per-pipeline metrics and render the table.

Metrics are **pooled per run**: for one pipeline and one repeat, counts are
summed over all documents (micro-average), giving one value per metric per
repeat. The table reports the mean and sample standard deviation across
repeats (the design doc runs each document 3 times). A metric with no
denominator (e.g. no judge, or no conflict triggered) is ``None`` and shown
as ``n/a``.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from eval.dataset import RECALL_TYPES
from eval.metrics import ConflictStatus, DocScore
from reviewdesk.contracts import utcnow

RATIO = "ratio"
SCORE = "score"
COUNT = "count"

METRICS: list[tuple[str, str, str]] = [
    *[(f"recall.{t.value}", f"Recall: {t.value}", RATIO) for t in RECALL_TYPES],
    ("recall.overall", "Recall: overall", RATIO),
    ("precision", "Precision (matched + judge)", RATIO),
    ("precision_matched", "Precision (matched only)", RATIO),
    ("citation_validity", "Citation validity (judge)", RATIO),
    ("citation_wellformed", "Citations well-formed", RATIO),
    ("rebuttal_strength", "Rebuttal strength (1-5)", SCORE),
    ("conflict_accuracy", "Conflict-resolution accuracy", RATIO),
    ("conflicts_triggered", "Conflicts triggered", RATIO),
    ("tokens_per_doc", "Tokens per doc", COUNT),
    ("search_calls_per_doc", "Search calls per doc", COUNT),
    ("llm_calls_per_doc", "LLM calls per doc", COUNT),
    ("seconds_per_doc", "Seconds per doc", SCORE),
    ("errors", "Failed runs", COUNT),
]
"""(key, label, kind) for every reported metric, in table order."""


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def run_metrics(scores: list[DocScore]) -> dict[str, float | None]:
    """Pooled metrics for one pipeline's scores in one repeat."""
    out: dict[str, float | None] = {}
    caught_all = total_all = 0
    for t in RECALL_TYPES:
        outcomes = [o for s in scores for o in s.defects if o.type is t]
        caught = sum(o.caught for o in outcomes)
        out[f"recall.{t.value}"] = _ratio(caught, len(outcomes))
        caught_all += caught
        total_all += len(outcomes)
    out["recall.overall"] = _ratio(caught_all, total_all)

    items = sum(s.items for s in scores)
    matched = sum(s.items_matched for s in scores)
    out["precision_matched"] = _ratio(matched, items)
    judged = all(s.judged for s in scores) and bool(scores)
    if judged:
        den = matched + sum(s.items_judged for s in scores)
        out["precision"] = _ratio(matched + sum(s.items_judged_real for s in scores), den)
        out["citation_validity"] = _ratio(
            sum(s.citations_supported for s in scores), sum(s.citations_judged for s in scores)
        )
    else:
        out["precision"] = None
        out["citation_validity"] = None
    out["citation_wellformed"] = _ratio(
        sum(s.citations_wellformed for s in scores), sum(s.citations for s in scores)
    )
    rebuttal_scores = [o.rebuttal_score for s in scores for o in s.defects if o.rebuttal_score]
    out["rebuttal_strength"] = statistics.fmean(rebuttal_scores) if rebuttal_scores else None

    conflicts = [c for s in scores for c in s.conflicts]
    resolved = sum(c.status is ConflictStatus.RESOLVED for c in conflicts)
    violated = sum(c.status is ConflictStatus.VIOLATED for c in conflicts)
    out["conflict_accuracy"] = _ratio(resolved, resolved + violated)
    out["conflicts_triggered"] = _ratio(resolved + violated, len(conflicts))

    n = len(scores)
    out["tokens_per_doc"] = _ratio(sum(s.usage.total_tokens for s in scores), n)
    out["search_calls_per_doc"] = _ratio(sum(s.usage.search_calls for s in scores), n)
    out["llm_calls_per_doc"] = _ratio(sum(s.usage.llm_calls for s in scores), n)
    out["seconds_per_doc"] = _ratio(sum(s.seconds for s in scores), n)
    out["errors"] = float(sum(s.error is not None for s in scores))
    return out


class Stat(BaseModel):
    """Mean and sample standard deviation of a metric across repeats."""

    model_config = ConfigDict(extra="forbid")

    mean: float | None
    stdev: float | None
    n: int
    """Repeats that had a value."""


def summarize(values: list[float | None]) -> Stat:
    """Mean ± stdev over the non-None values (stdev None with fewer than 2)."""
    present = [v for v in values if v is not None]
    if not present:
        return Stat(mean=None, stdev=None, n=0)
    stdev = statistics.stdev(present) if len(present) > 1 else None
    return Stat(mean=statistics.fmean(present), stdev=stdev, n=len(present))


class EvalResults(BaseModel):
    """Everything one harness invocation produced."""

    model_config = ConfigDict(extra="forbid")

    pipelines: list[str]
    doc_ids: list[str]
    repeats: int
    judged: bool
    started_at: datetime = Field(default_factory=utcnow)
    scores: list[DocScore] = Field(default_factory=list)

    def per_run(self) -> dict[str, list[dict[str, float | None]]]:
        """Pipeline → one metrics dict per repeat."""
        grouped: dict[tuple[str, int], list[DocScore]] = defaultdict(list)
        for s in self.scores:
            grouped[(s.pipeline, s.repeat)].append(s)
        return {
            p: [run_metrics(grouped[(p, r)]) for r in range(self.repeats) if grouped[(p, r)]]
            for p in self.pipelines
        }

    def summary(self) -> dict[str, dict[str, Stat]]:
        """Pipeline → metric key → ``Stat`` across repeats."""
        return {
            p: {key: summarize([run[key] for run in runs]) for key, _, _ in METRICS}
            for p, runs in self.per_run().items()
        }

    def table(self) -> str:
        """Markdown results table: one row per metric, one column per pipeline."""
        return render_table(self)


def _fmt(stat: Stat, kind: str) -> str:
    if stat.mean is None:
        return "n/a"
    if kind in (RATIO, SCORE):
        text = f"{stat.mean:.2f}"
        spread = f" ± {stat.stdev:.2f}" if stat.stdev is not None else ""
    else:
        text = f"{stat.mean:,.0f}"
        spread = f" ± {stat.stdev:,.0f}" if stat.stdev is not None else ""
    return text + spread


def render_table(results: EvalResults) -> str:
    """Render ``results`` as a markdown table with a one-line caption."""
    summary = results.summary()
    header = ["Metric", *results.pipelines]
    rows = [
        [label, *(_fmt(summary[p][key], kind) for p in results.pipelines)]
        for key, label, kind in METRICS
    ]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    caption = (
        f"{len(results.doc_ids)} documents x {results.repeats} repeat(s); "
        f"judge {'on' if results.judged else 'off'}; mean ± stdev across repeats."
    )
    return "\n".join([caption, "", line(header), sep, *(line(r) for r in rows)])
