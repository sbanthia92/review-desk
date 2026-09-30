"""Claim extractor agent (T3).

Splits a document into thesis, supporting and factual claims with exact spans
and importance scores, returned as ``AddClaim`` ledger updates.
"""

from reviewdesk.agents.extractor.agent import ExtractionOutcome, ExtractorAgent
from reviewdesk.agents.extractor.prompts import TAG_EXTRACT, ExtractedClaim, ExtractionOutput
from reviewdesk.agents.extractor.spans import Located, MatchMethod, TextLocator

__all__ = [
    "TAG_EXTRACT",
    "ExtractedClaim",
    "ExtractionOutcome",
    "ExtractionOutput",
    "ExtractorAgent",
    "Located",
    "MatchMethod",
    "TextLocator",
]
