"""Seeded documents, answer keys and the dataset loader.

A seeded document is an original, invented text with known defects. Its answer
key lists each defect with its type, the exact quoted text (located to a
``Span`` on load) and type-specific expectations. On disk a document is two
files in ``eval/data/``: ``<id>.md`` (the text) and ``<id>.json`` (the key).
The JSON may carry the text inline under ``"text"`` instead.

Every span is validated against the text when the key is loaded: a quote must
occur in the text (exactly), an ambiguous quote needs ``occurrence``, and an
explicit ``span`` must cover exactly the quote.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from reviewdesk.contracts import Document, Profile, Severity, Span, Verdict

DATA_DIR = Path(__file__).parent / "data"
"""Default dataset directory."""


class DefectType(StrEnum):
    """Kind of seeded defect, and which agent is expected to catch it."""

    WRONG_FACT = "wrong_fact"
    """Wrong number, date or attribution (fact-checker, verdict wrong)."""
    UNSUPPORTED = "unsupported"
    """A checkable claim no source supports (fact-checker, verdict unsupported)."""
    WEAK_ARGUMENT = "weak_argument"
    """A deliberately weak argument with a known strong rebuttal (devil's advocate)."""
    BORROWED_SENTENCE = "borrowed_sentence"
    """A sentence copied from a known source (originality checker)."""
    GRAMMAR = "grammar"
    """A grammar or clarity error (copy editor)."""
    CONFLICT = "conflict"
    """A seeded conflict between agents, settled by one of the design's rules."""


RECALL_TYPES: tuple[DefectType, ...] = (
    DefectType.WRONG_FACT,
    DefectType.UNSUPPORTED,
    DefectType.WEAK_ARGUMENT,
    DefectType.BORROWED_SENTENCE,
    DefectType.GRAMMAR,
)
"""Defect types scored by recall (conflicts are scored separately)."""


class ConflictRule(StrEnum):
    """The design doc's conflict-resolution rules that a seeded conflict tests."""

    COPYEDIT_YIELDS_TO_FACT = "copyedit_yields_to_fact"
    """A copy edit touching a span flagged wrong/unsupported is dropped."""
    REBUTTAL_ON_VERIFIED_FACT = "rebuttal_on_verified_fact"
    """A rebuttal of a verified claim survives only if it targets the interpretation."""
    SAME_SPAN_MERGE = "same_span_merge"
    """Two agents flagging the same span are merged into one finding."""
    EVIDENCE_FREE_REBUTTAL = "evidence_free_rebuttal"
    """A rebuttal with no retrieved source is downgraded to ``consider``."""


DEFAULT_SEVERITY: dict[DefectType, Severity] = {
    DefectType.WRONG_FACT: Severity.FACTUAL_ERROR,
    DefectType.UNSUPPORTED: Severity.UNSUPPORTED,
    DefectType.WEAK_ARGUMENT: Severity.STRONG_REBUTTAL,
    DefectType.BORROWED_SENTENCE: Severity.CONSIDER,
    DefectType.GRAMMAR: Severity.STYLE,
}
"""Expected finding severity per defect type when the key does not say."""

DEFAULT_VERDICT: dict[DefectType, Verdict] = {
    DefectType.WRONG_FACT: Verdict.WRONG,
    DefectType.UNSUPPORTED: Verdict.UNSUPPORTED,
}
"""Expected fact-check verdict for fact defects when the key does not say."""


class Genre(StrEnum):
    """Rough kind of document, for slicing results."""

    FOOTBALL_BLOG = "football_blog"
    OP_ED = "op_ed"
    TECH_BLOG = "tech_blog"
    DESIGN_DOC = "design_doc"


class SeededDefect(BaseModel):
    """One seeded defect in an answer key.

    ``quote`` is the exact defective text; ``span`` is filled from it on load
    (or given explicitly, in which case it must cover ``quote`` exactly).
    ``occurrence`` (1-based) picks among repeated quotes.

    Type-specific fields:

    - ``wrong_fact`` / ``unsupported``: ``expected_verdict`` (defaulted);
      ``correction`` is the true fact for wrong facts (optional, used by fake
      pipelines and judges).
    - ``weak_argument``: ``known_rebuttal`` (required): the strongest known
      rebuttal, scored against by the rebuttal-strength judge.
    - ``borrowed_sentence``: ``source_url`` (required) and ``source_title``.
    - ``grammar``: ``correction`` (optional) is the fixed text.
    - ``conflict``: ``rule`` (required).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    type: DefectType
    quote: str = Field(min_length=1)
    occurrence: int | None = Field(default=None, ge=1)
    span: Span | None = None
    expected_severity: Severity | None = None
    expected_verdict: Verdict | None = None
    correction: str | None = None
    known_rebuttal: str | None = None
    source_url: str | None = None
    source_title: str | None = None
    rule: ConflictRule | None = None
    note: str = ""

    @model_validator(mode="after")
    def _type_fields(self) -> SeededDefect:
        t = self.type
        if t is DefectType.WEAK_ARGUMENT and not self.known_rebuttal:
            raise ValueError(f"defect {self.id}: weak_argument needs known_rebuttal")
        if t is DefectType.BORROWED_SENTENCE and not self.source_url:
            raise ValueError(f"defect {self.id}: borrowed_sentence needs source_url")
        if self.source_url and not self.source_url.startswith(("http://", "https://")):
            raise ValueError(f"defect {self.id}: source_url must be http(s)")
        if (t is DefectType.CONFLICT) != (self.rule is not None):
            raise ValueError(f"defect {self.id}: rule is required for, and only for, conflicts")
        if self.expected_severity is None and t in DEFAULT_SEVERITY:
            self.expected_severity = DEFAULT_SEVERITY[t]
        if self.expected_verdict is None and t in DEFAULT_VERDICT:
            self.expected_verdict = DEFAULT_VERDICT[t]
        return self

    @property
    def located(self) -> Span:
        """The validated span. Only valid on defects of a loaded document."""
        if self.span is None:
            raise ValueError(f"defect {self.id} has not been located")
        return self.span


class SeededDocument(BaseModel):
    """A seeded document plus its answer key.

    Building one validates and locates every defect span against ``text``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str
    genre: Genre
    profile: Profile
    text: str = Field(min_length=1)
    defects: list[SeededDefect] = Field(default_factory=list)

    @model_validator(mode="after")
    def _locate(self) -> SeededDocument:
        if self.profile is Profile.AUTO:
            raise ValueError(f"{self.id}: answer keys need a concrete profile")
        ids = [d.id for d in self.defects]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"{self.id}: duplicate defect ids {dupes}")
        for defect in self.defects:
            defect.span = locate_defect(self.text, defect, doc_id=self.id)
        return self

    @property
    def document(self) -> Document:
        """The document to review (its ``id`` is the seeded document's id)."""
        return Document.from_text(self.text, id=self.id)

    def of_type(self, *types: DefectType) -> list[SeededDefect]:
        """Defects of the given types, in document order."""
        found = [d for d in self.defects if d.type in types]
        return sorted(found, key=lambda d: d.located.start)

    def defect(self, defect_id: str) -> SeededDefect:
        """The defect with this id (``KeyError`` if absent)."""
        for d in self.defects:
            if d.id == defect_id:
                return d
        raise KeyError(defect_id)


class DatasetError(ValueError):
    """A dataset file is malformed or its answer key does not match its text."""


def locate_defect(text: str, defect: SeededDefect, *, doc_id: str = "") -> Span:
    """Locate ``defect.quote`` in ``text`` and return its span.

    Raises ``ValueError`` if the quote is absent, ambiguous without an
    ``occurrence``, or disagrees with an explicit ``span``.
    """
    where = f"{doc_id}/{defect.id}" if doc_id else defect.id
    starts = _occurrences(text, defect.quote)
    if not starts:
        raise ValueError(f"{where}: quote not found in text: {defect.quote[:60]!r}")
    if defect.span is not None:
        if defect.span.text_of(text) != defect.quote or not defect.span.is_valid_for(text):
            raise ValueError(f"{where}: span does not cover the quote exactly")
        return defect.span
    if defect.occurrence is None:
        if len(starts) > 1:
            raise ValueError(f"{where}: quote occurs {len(starts)} times; set occurrence")
        start = starts[0]
    else:
        if defect.occurrence > len(starts):
            raise ValueError(f"{where}: occurrence {defect.occurrence} > {len(starts)}")
        start = starts[defect.occurrence - 1]
    return Span(start=start, end=start + len(defect.quote))


def _occurrences(text: str, quote: str) -> list[int]:
    starts: list[int] = []
    i = text.find(quote)
    while i != -1:
        starts.append(i)
        i = text.find(quote, i + 1)
    return starts


def load_document(key_path: Path) -> SeededDocument:
    """Load one answer key (``<id>.json``) and its text (``<id>.md``)."""
    try:
        raw: dict[str, Any] = json.loads(key_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"{key_path.name}: cannot read answer key: {exc}") from exc
    if "text" not in raw:
        text_path = key_path.with_suffix(".md")
        try:
            raw["text"] = text_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DatasetError(f"{key_path.name}: missing text file {text_path.name}") from exc
    raw.setdefault("id", key_path.stem)
    if raw["id"] != key_path.stem:
        raise DatasetError(f"{key_path.name}: id {raw['id']!r} must match the file name")
    try:
        return SeededDocument.model_validate(raw)
    except ValidationError as exc:
        raise DatasetError(f"{key_path.name}: {exc}") from exc


def load_dataset(
    directory: Path | None = None, *, ids: list[str] | None = None
) -> list[SeededDocument]:
    """Load every seeded document in ``directory`` (default ``eval/data``), by id.

    ``ids`` restricts the result to those documents (unknown ids raise).
    """
    root = directory or DATA_DIR
    docs = [load_document(p) for p in sorted(root.glob("*.json"))]
    if ids is not None:
        by_id = {d.id: d for d in docs}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise DatasetError(f"unknown document ids: {missing}")
        docs = [by_id[i] for i in ids]
    return docs
