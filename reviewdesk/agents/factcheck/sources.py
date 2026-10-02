"""Deterministic helpers for handling retrieved sources.

Everything here treats page text as untrusted data: it is searched, sliced
and quoted, never interpreted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlsplit

from reviewdesk.agents.factcheck.schemas import Stance
from reviewdesk.contracts import Evidence, utcnow

MAX_PASSAGES = 5
"""Passages kept per page by ``find_in_page``."""

MAX_PASSAGE_CHARS = 400
MAX_EXCERPT_CHARS = 500

_WORD = re.compile(r"[a-z0-9][a-z0-9.,%-]*", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORD_TEXT = (
    "a an and are as at be by for from has have in is it its of on or that the their "
    "there this to was were with than any other more most"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

_PRIMARY_HOST_SUFFIXES = (".gov", ".edu", ".int", ".mil")
_PRIMARY_HOSTS = frozenset(
    {"doi.org", "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "data.worldbank.org", "sec.gov"}
)

_LOW_QUALITY_HOSTS = frozenset(
    {
        "facebook.com",
        "instagram.com",
        "youtube.com",
        "youtu.be",
        "tiktok.com",
        "x.com",
        "twitter.com",
        "reddit.com",
        "pinterest.com",
        "quora.com",
        "linkedin.com",
        "threads.net",
    }
)
"""Social and user-generated hosts. Shown as evidence, but they never count
towards settling a claim."""

_REPUTABLE_HOSTS = frozenset(
    {
        "wikipedia.org",
        "britannica.com",
        "bbc.com",
        "bbc.co.uk",
        "reuters.com",
        "apnews.com",
        "nytimes.com",
        "theguardian.com",
        "washingtonpost.com",
        "ft.com",
        "economist.com",
        "espn.com",
        "skysports.com",
        "uefa.com",
        "fifa.com",
        "premierleague.com",
        "history.com",
        "nature.com",
        "science.org",
        "who.int",
        "un.org",
    }
)
"""Edited reference works, major news organisations and official bodies. One
of these is enough to verify a claim nothing contradicts."""

_INJECTION = re.compile(
    r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions"
    r"|disregard\s+(all\s+|the\s+)?(previous|prior|above)"
    r"|(mark|label|rate|classify)\s+(this|the)\s+(claim\s+)?(as\s+)?(verified|true|correct)"
    r"|you\s+are\s+now\s+"
    r"|system\s+prompt"
    r"|as\s+an?\s+(ai|language\s+model|fact-?checker)[, ]+you\s+must",
    re.IGNORECASE,
)


def domain(url: str) -> str:
    """Host of ``url`` without a leading ``www.`` (used for source independence)."""
    host = (urlsplit(url).hostname or url).lower()
    return host.removeprefix("www.")


def _host_in(url: str, hosts: frozenset[str]) -> bool:
    host = domain(url)
    return any(host == h or host.endswith("." + h) for h in hosts)


def is_low_quality(url: str) -> bool:
    """True for social and user-generated hosts."""
    return _host_in(url, _LOW_QUALITY_HOSTS)


def is_reputable(url: str) -> bool:
    """True for edited reference works, major news and official bodies."""
    host = domain(url)
    return (
        _host_in(url, _REPUTABLE_HOSTS)
        or host in _PRIMARY_HOSTS
        or host.endswith(_PRIMARY_HOST_SUFFIXES)
    )


def is_http_url(url: str) -> bool:
    """True for absolute http(s) URLs."""
    parts = urlsplit(url)
    return parts.scheme in {"http", "https"} and bool(parts.hostname)


def primary_hint(url: str) -> bool:
    """Cheap guess that a URL is a primary source (used only to order fetches)."""
    host = domain(url)
    path = urlsplit(url).path.lower()
    return (
        host in _PRIMARY_HOSTS
        or host.endswith(_PRIMARY_HOST_SUFFIXES)
        or host.startswith("data.")
        or path.endswith(".pdf")
    )


def looks_like_injection(text: str) -> bool:
    """True if the text contains instructions aimed at the reviewer.

    Such pages are dropped as evidence: a source that tries to steer the
    fact-checker is not trustworthy evidence of anything.
    """
    return bool(_INJECTION.search(text))


_TERM_PREFIX = 6
"""Words are compared on their first letters, so spelling and inflection
variants match ("refueling"/"refuelling", "charter"/"chartered")."""


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in _WORD.findall(text):
        word = raw.strip(".,-").lower()
        if len(word) <= 1 or word in _STOPWORDS:
            continue
        terms.add(word if any(ch.isdigit() for ch in word) else word[:_TERM_PREFIX])
    return terms


def find_in_page(text: str, claim: str, query: str = "") -> list[str]:
    """Return up to ``MAX_PASSAGES`` sentences of ``text`` most relevant to the claim.

    Scores sentences by term overlap with the claim (and query); numbers count
    double since factual claims usually hinge on them. Passages keep page
    order so context reads naturally.
    """
    wanted = _terms(claim) | _terms(query)
    if not wanted:
        return []
    numbers = {t for t in wanted if any(ch.isdigit() for ch in t)}
    scored: list[tuple[float, int, str]] = []
    for index, raw in enumerate(_SENTENCE.split(text)):
        sentence = " ".join(raw.split())
        if not sentence:
            continue
        terms = _terms(sentence)
        score = len(terms & wanted) + len(terms & numbers)
        if score > 0:
            scored.append((score, index, sentence[:MAX_PASSAGE_CHARS]))
    best = sorted(scored, key=lambda s: (-s[0], s[1]))[:MAX_PASSAGES]
    return [s for _, _, s in sorted(best, key=lambda s: s[1])]


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def verified_quote(quote: str, source_text: str, fallback: str) -> str:
    """``quote`` if it appears verbatim in ``source_text``, else ``fallback``.

    Keeps evidence excerpts honest: an excerpt is always text we retrieved.
    """
    quote = " ".join(quote.split())
    if quote and _norm(quote) in _norm(source_text):
        return quote[:MAX_EXCERPT_CHARS]
    return " ".join(fallback.split())[:MAX_EXCERPT_CHARS]


@dataclass
class Source:
    """One retrieved source the loop has read (a fetched page or a snippet)."""

    url: str
    title: str
    text: str
    """The retrieved text excerpts are validated against (untrusted)."""
    passages: list[str]
    """What the model is shown (a few relevant sentences)."""
    kind: str = "page"
    """``page``, ``snippet``, ``prior-evidence`` or ``counter-evidence``."""
    can_be_primary: bool = True
    """False when we only saw a snippet, so we cannot confirm it is primary."""
    suspicious: bool = False
    retrieved_at: datetime = field(default_factory=utcnow)
    stance: Stance | None = None
    evidence: Evidence | None = None

    @property
    def domain(self) -> str:
        """Host used to judge independence between sources."""
        return domain(self.url)


def relevant(sources: list[Source], stance: Stance) -> list[Source]:
    """Assessed sources with ``stance`` and evidence, primary sources first."""
    hits = [s for s in sources if s.stance is stance and s.evidence is not None]
    return sorted(hits, key=lambda s: not (s.evidence and s.evidence.is_primary))


def settles(sources: list[Source], stance: Stance) -> bool:
    """True if a primary source, or two independent sources, take ``stance``.

    Social and user-generated hosts do not count towards the two.
    """
    hits = relevant(sources, stance)
    if any(s.evidence is not None and s.evidence.is_primary for s in hits):
        return True
    return len({s.domain for s in hits if not is_low_quality(s.url)}) >= 2


def reputably_supported(sources: list[Source]) -> bool:
    """True if a reputable source supports the claim and nothing contradicts it.

    The relaxed rule for uncontested facts: one edited, reputable source that
    was actually fetched (not just a search snippet) is enough when no
    retrieved source disagrees.
    """
    if relevant(sources, Stance.CONTRADICTS):
        return False
    # ``can_be_primary`` is False for search snippets: the page itself must have been read.
    return any(is_reputable(s.url) and s.can_be_primary for s in relevant(sources, Stance.SUPPORTS))


def shared_terms(claim: str, text: str) -> int:
    """How many of the claim's terms appear in ``text`` (numbers count double)."""
    wanted = _terms(claim)
    found = wanted & _terms(text)
    return len(found) + sum(1 for t in found if any(ch.isdigit() for ch in t))
