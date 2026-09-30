"""Structure reviewer agent (T6).

Flow, argument-order and missing-section findings (``Severity.STRUCTURE``);
the design-doc variant checks gaps and unstated assumptions.
"""

from reviewdesk.agents.structure.agent import StructureAgent, StructureOutcome, build_outline
from reviewdesk.agents.structure.prompts import (
    TAG_DESIGN_DOC,
    TAG_REVIEW,
    IssueKind,
    StructureIssue,
    StructureOutput,
)

__all__ = [
    "TAG_DESIGN_DOC",
    "TAG_REVIEW",
    "IssueKind",
    "StructureAgent",
    "StructureIssue",
    "StructureOutcome",
    "StructureOutput",
    "build_outline",
]
