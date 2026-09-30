"""Human-readable agent labels and the deterministic agent order."""

from __future__ import annotations

from reviewdesk.contracts import AgentName

AGENT_LABELS: dict[str, str] = {
    AgentName.EXTRACTOR: "claim extraction",
    AgentName.FACTCHECK: "fact-check",
    AgentName.DEVILS_ADVOCATE: "devil's advocate",
    AgentName.COPYEDIT: "copy edit",
    AgentName.STRUCTURE: "structure review",
    AgentName.ORIGINALITY: "originality check",
    AgentName.ORCHESTRATOR: "orchestrator",
}

AGENT_ORDER: tuple[AgentName, ...] = (
    AgentName.FACTCHECK,
    AgentName.DEVILS_ADVOCATE,
    AgentName.STRUCTURE,
    AgentName.COPYEDIT,
    AgentName.ORIGINALITY,
    AgentName.EXTRACTOR,
    AgentName.ORCHESTRATOR,
)
"""Order results are applied in, and the tie-break order for merged findings."""


def label(name: str) -> str:
    """User-facing label for an agent name (falls back to the name)."""
    return AGENT_LABELS.get(name, name.replace("_", " "))


def agent_rank(name: str) -> int:
    """Position of ``name`` in ``AGENT_ORDER`` (unknown agents sort last)."""
    for index, known in enumerate(AGENT_ORDER):
        if known == name:
            return index
    return len(AGENT_ORDER)
