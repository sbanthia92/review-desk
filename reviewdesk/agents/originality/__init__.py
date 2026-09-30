"""Originality checker agent (T7).

Samples distinctive sentences, searches for near-matches and reports possible
matches with URLs. Every finding is heuristic (``Finding.heuristic``) with
``Severity.CONSIDER``; no ledger updates.
"""

from reviewdesk.agents.originality.agent import (
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_THRESHOLD,
    OriginalityAgent,
    build_query,
)
from reviewdesk.agents.originality.sentences import Sentence, distinctive_sentences, split_sentences
from reviewdesk.agents.originality.similarity import Match, normalize_url, same_url, similarity

__all__ = [
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_THRESHOLD",
    "Match",
    "OriginalityAgent",
    "Sentence",
    "build_query",
    "distinctive_sentences",
    "normalize_url",
    "same_url",
    "similarity",
    "split_sentences",
]
