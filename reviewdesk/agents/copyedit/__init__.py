"""Copy editor agent (T6).

Line-level grammar, clarity and concision suggestions, each a
``Severity.STYLE`` finding with a span into the document and a suggestion.
"""

from reviewdesk.agents.copyedit.agent import CopyEditAgent, EditOutcome
from reviewdesk.agents.copyedit.prompts import TAG_EDIT, CopyEditOutput, ProposedEdit

__all__ = ["TAG_EDIT", "CopyEditAgent", "CopyEditOutput", "EditOutcome", "ProposedEdit"]
