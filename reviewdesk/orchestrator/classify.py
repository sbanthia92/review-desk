"""Profile classification: a cheap-tier LLM call with a heuristic fallback."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from reviewdesk.contracts import Document, LLMClient, ModelTier, Profile, Usage
from reviewdesk.orchestrator import prompts
from reviewdesk.orchestrator.schemas import ClassifyReply

log = logging.getLogger(__name__)

_DESIGN_HEADINGS = re.compile(
    r"^\s{0,3}#{1,6}\s*(design|proposal|context|background|goals|non-goals|alternatives"
    r"( considered)?|rollout|migration|architecture|requirements|open questions|risks"
    r"|implementation|api|motivation|overview)\b",
    re.IGNORECASE | re.MULTILINE,
)
_DESIGN_TERMS = re.compile(
    r"\b(we propose|this proposal|rollout|migration|latency|throughput|schema|endpoint"
    r"|service|database|deploy(ment)?|RFC|trade-?offs?|alternatives considered)\b",
    re.IGNORECASE,
)


def heuristic_profile(doc: Document) -> Profile:
    """Guess the profile from headings and vocabulary (the no-LLM fallback)."""
    text = doc.text
    headings = len(_DESIGN_HEADINGS.findall(text))
    if re.match(r"\s*#\s*(design|rfc|proposal)\b", text, re.IGNORECASE):
        headings += 2
    terms = len(_DESIGN_TERMS.findall(text))
    words = max(1, doc.word_count)
    score = headings * 2 + min(terms, 20) * (200 / max(words, 200))
    return Profile.DESIGN_DOC if score >= 4 else Profile.OPINION


@dataclass(frozen=True)
class Classification:
    """The chosen profile, how it was chosen, and the LLM usage spent."""

    profile: Profile
    method: str  # "user", "llm" or "heuristic"
    usage: Usage


async def classify(
    doc: Document, profile: Profile, llm: LLMClient, *, time_limit: float | None = None
) -> Classification:
    """Resolve ``profile``: user choice, else the LLM, else the heuristic.

    Never raises: any LLM failure (provider error, bad schema, timeout) falls
    back to ``heuristic_profile``.
    """
    if profile is not Profile.AUTO:
        return Classification(profile, "user", Usage())
    try:
        async with asyncio.timeout(time_limit):
            response = await llm.complete(
                prompts.classify_messages(doc),
                schema=ClassifyReply,
                model_tier=ModelTier.CHEAP,
                tag=prompts.TAG_CLASSIFY,
                max_tokens=200,
            )
    except Exception as exc:
        log.info("orchestrator: classification fell back to heuristic (%s)", type(exc).__name__)
        return Classification(heuristic_profile(doc), "heuristic", Usage())
    reply = response.parsed
    if not isinstance(reply, ClassifyReply):
        return Classification(heuristic_profile(doc), "heuristic", response.usage)
    return Classification(Profile(reply.profile), "llm", response.usage)
