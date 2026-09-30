"""Pipeline entry point: ``run_review`` (implements ``ReviewPipeline``).

The pipeline composes agents from an ``AgentRegistry``. Pass one explicitly,
or install a process-wide default once at startup (T11's wiring does this)::

    from reviewdesk.pipeline import run_review, set_default_registry
    set_default_registry(registry)
    report = await run_review(doc, Profile.AUTO, on_progress)
"""

from __future__ import annotations

from reviewdesk.contracts import (
    Budget,
    Document,
    MissingSetup,
    Profile,
    ProgressCallback,
    Report,
)
from reviewdesk.orchestrator import AgentRegistry, Orchestrator

__all__ = [
    "AgentRegistry",
    "Orchestrator",
    "get_default_registry",
    "run_review",
    "set_default_registry",
]

_default_registry: AgentRegistry | None = None


def set_default_registry(registry: AgentRegistry | None) -> None:
    """Install (or clear, with None) the registry ``run_review`` uses by default."""
    global _default_registry
    _default_registry = registry


def get_default_registry() -> AgentRegistry:
    """The default registry; raises ``MissingSetup`` if none is installed."""
    if _default_registry is None:
        raise MissingSetup(
            "Review Desk is not set up: no agent registry is configured "
            "(call reviewdesk.pipeline.set_default_registry)."
        )
    return _default_registry


async def run_review(
    doc: Document,
    profile: Profile,
    on_progress: ProgressCallback,
    *,
    registry: AgentRegistry | None = None,
    budget: Budget | None = None,
    focus: str | None = None,
    style_guide: str | None = None,
) -> Report:
    """Review ``doc`` and return the structured report.

    ``profile`` may be ``Profile.AUTO`` (classified by the orchestrator).
    Progress goes to ``on_progress`` as one monotonic 0-100 sequence through
    classifying, extracting, planning, reviewing, reacting, resolving,
    reporting and done. Agent failures and time-outs degrade the report
    (see ``Report.notes``) instead of raising. Raises ``MissingSetup`` when no
    registry is passed and no default is installed.
    """
    chosen = registry if registry is not None else get_default_registry()
    orchestrator = Orchestrator(chosen, budget=budget)
    return await orchestrator.review(
        doc, profile, on_progress, focus=focus, style_guide=style_guide
    )
