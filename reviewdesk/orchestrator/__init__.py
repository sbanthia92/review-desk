"""Orchestrator: classification, planning, parallel review, reactions, conflict rules, report.

Agents come from an ``AgentRegistry``; nothing here imports a concrete agent.
"""

from reviewdesk.orchestrator.assemble import build_report, count_by_severity, verdict_line
from reviewdesk.orchestrator.classify import Classification, classify, heuristic_profile
from reviewdesk.orchestrator.conflicts import (
    Decision,
    Resolution,
    discard_fact_rebuttals_on_verified,
    downgrade_unsourced_rebuttals,
    drop_copy_edits_on_flagged_spans,
    ensure_rebuttal_findings,
    merge_same_span_findings,
    rank_findings,
    rank_rebuttals,
    resolve_conflicts,
)
from reviewdesk.orchestrator.orchestrator import Orchestrator
from reviewdesk.orchestrator.planning import default_plan, split_budget, validate_plan
from reviewdesk.orchestrator.progress import STEP_ORDER, STEP_RANGES, ProgressTracker
from reviewdesk.orchestrator.prompts import TAG_CLASSIFY, TAG_PLAN, TAG_RULING
from reviewdesk.orchestrator.registry import AgentRegistry

__all__ = [
    "STEP_ORDER",
    "STEP_RANGES",
    "TAG_CLASSIFY",
    "TAG_PLAN",
    "TAG_RULING",
    "AgentRegistry",
    "Classification",
    "Decision",
    "Orchestrator",
    "ProgressTracker",
    "Resolution",
    "build_report",
    "classify",
    "count_by_severity",
    "default_plan",
    "discard_fact_rebuttals_on_verified",
    "downgrade_unsourced_rebuttals",
    "drop_copy_edits_on_flagged_spans",
    "ensure_rebuttal_findings",
    "heuristic_profile",
    "merge_same_span_findings",
    "rank_findings",
    "rank_rebuttals",
    "resolve_conflicts",
    "split_budget",
    "validate_plan",
    "verdict_line",
]
