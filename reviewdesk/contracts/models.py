"""Shared data models for Review Desk.

Every agent, the orchestrator, the report renderer and the eval harness code
against these types. They are frozen by Wave 0: request changes through
``CONTRACT_CHANGES.md`` instead of editing this file from a task branch.

Conventions:

- Spans are character offsets into ``Document.text``: ``start`` inclusive,
  ``end`` exclusive, so ``doc.text[span.start:span.end]`` is the quoted text.
- IDs are short opaque strings. ``new_id(prefix)`` makes a unique one.
- Agents never mutate the shared ledger directly. They return
  ``LedgerUpdate``s in their ``AgentResult`` and the orchestrator applies them
  with ``ClaimLedger.apply``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def new_id(prefix: str) -> str:
    """Return a unique ID such as ``claim_3f9a1c2b``."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class _Model(BaseModel):
    """Base for all contract models: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Documents and spans
# ---------------------------------------------------------------------------


class Document(_Model):
    """A document under review.

    ``text`` is the full plain text or markdown. ``source_url`` is set when the
    document was fetched from a URL. ``word_count`` is whitespace-split words.
    """

    id: str = Field(default_factory=lambda: new_id("doc"))
    text: str
    source_url: str | None = None
    word_count: int = Field(ge=0)

    @classmethod
    def from_text(
        cls, text: str, *, id: str | None = None, source_url: str | None = None
    ) -> Document:
        """Build a document from text, computing ``word_count``."""
        return cls(
            id=id or new_id("doc"),
            text=text,
            source_url=source_url,
            word_count=len(text.split()),
        )


class Span(_Model):
    """A half-open character range ``[start, end)`` into ``Document.text``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_order(self) -> Span:
        if self.end < self.start:
            raise ValueError(f"span end {self.end} is before start {self.start}")
        return self

    def overlaps(self, other: Span) -> bool:
        """True if the two spans share at least one character.

        Empty spans overlap nothing.
        """
        if self.start == self.end or other.start == other.end:
            return False
        return self.start < other.end and other.start < self.end

    def contains(self, other: Span) -> bool:
        """True if ``other`` lies entirely inside this span."""
        return self.start <= other.start and other.end <= self.end

    def text_of(self, text: str) -> str:
        """Return the slice of ``text`` this span covers."""
        return text[self.start : self.end]

    def is_valid_for(self, text: str) -> bool:
        """True if the span lies within ``text``."""
        return self.end <= len(text)


# ---------------------------------------------------------------------------
# Claims, evidence and verdicts
# ---------------------------------------------------------------------------


class ClaimType(StrEnum):
    """Kind of claim found by the extractor."""

    THESIS = "thesis"
    """The document's central argument. Usually one per document."""
    SUPPORTING = "supporting"
    """An argumentative claim that backs the thesis."""
    FACTUAL = "factual"
    """A checkable fact: a number, date, attribution or event."""


class Claim(_Model):
    """One claim extracted from the document.

    ``text`` must equal ``span.text_of(document.text)`` exactly.
    ``importance`` is 0.0 (trivial) to 1.0 (the thesis hinges on it).
    ``standalone`` restates the claim so it can be understood (and searched
    for) without the rest of the document: pronouns and references such as
    "they" or "that season" are replaced by what they refer to. It is written
    by the extractor's model, so treat it as untrusted data like ``text``.
    Empty when the quote already stands alone.
    """

    id: str = Field(default_factory=lambda: new_id("claim"))
    text: str
    span: Span
    type: ClaimType
    importance: float = Field(ge=0.0, le=1.0)
    standalone: str = ""

    @property
    def checkable_text(self) -> str:
        """The claim as it should be researched: ``standalone`` if set, else ``text``."""
        return self.standalone or self.text


class Evidence(_Model):
    """A retrieved source excerpt backing a verdict, rebuttal or finding.

    ``excerpt`` is quoted from the fetched page (or the search snippet when the
    page was not fetched). ``is_primary`` marks a primary source (the original
    dataset, filing, transcript or publication) as opposed to reporting on it.
    """

    url: str
    title: str
    excerpt: str
    retrieved_at: datetime = Field(default_factory=utcnow)
    is_primary: bool = False


class Verdict(StrEnum):
    """Fact-check outcome for a claim."""

    VERIFIED = "verified"
    """Retrieved evidence supports the claim. Never set without evidence."""
    WRONG = "wrong"
    """Retrieved evidence contradicts the claim."""
    UNSUPPORTED = "unsupported"
    """No evidence settles the claim within the research budget."""
    UNCHECKED = "unchecked"
    """Not checked (skipped by the plan or out of budget)."""


class RebuttalTarget(StrEnum):
    """What a rebuttal attacks. Used by the conflict rules."""

    INTERPRETATION = "interpretation"
    """The reasoning, framing or conclusion drawn from the claim."""
    FACT = "fact"
    """The factual content of the claim itself."""


class Rebuttal(_Model):
    """An opposing argument from the devil's advocate.

    Every rebuttal targets at least one claim ID. A rebuttal with empty
    ``evidence`` is allowed but is downgraded to ``Severity.CONSIDER`` by the
    orchestrator. ``strength`` is 1 (weak) to 5 (devastating).
    """

    id: str = Field(default_factory=lambda: new_id("reb"))
    target_claim_ids: list[str] = Field(min_length=1)
    argument: str
    evidence: list[Evidence] = Field(default_factory=list)
    strength: int = Field(ge=1, le=5)
    target: RebuttalTarget = RebuttalTarget.INTERPRETATION

    @property
    def has_evidence(self) -> bool:
        """True if the rebuttal cites at least one retrieved source."""
        return bool(self.evidence)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class Severity(IntEnum):
    """Finding severity. Higher value ranks first in the report.

    Order: factual_error > unsupported > strong_rebuttal > structure > style >
    consider. ``CONSIDER`` is where evidence-free rebuttals are downgraded to.
    Use ``Severity.label`` for the snake_case name used in reports and JSON.
    """

    CONSIDER = 0
    STYLE = 1
    STRUCTURE = 2
    STRONG_REBUTTAL = 3
    UNSUPPORTED = 4
    FACTUAL_ERROR = 5

    @property
    def label(self) -> str:
        """Snake_case name, e.g. ``"factual_error"``."""
        return self.name.lower()


class AgentName(StrEnum):
    """Stable names of the specialist agents."""

    EXTRACTOR = "extractor"
    FACTCHECK = "factcheck"
    DEVILS_ADVOCATE = "devils_advocate"
    COPYEDIT = "copyedit"
    STRUCTURE = "structure"
    ORIGINALITY = "originality"
    ORCHESTRATOR = "orchestrator"


class Finding(_Model):
    """One reportable issue.

    ``agent`` is the producing agent's name (an ``AgentName`` value); merged
    findings list every contributor in ``merged_from``. ``span`` is None for
    document-level findings (e.g. a missing section). ``claim_ids`` links the
    finding to ledger entries. ``heuristic`` is True for findings that are
    best-effort guesses (every originality finding).
    """

    id: str = Field(default_factory=lambda: new_id("find"))
    agent: str
    severity: Severity
    span: Span | None = None
    message: str
    evidence: list[Evidence] = Field(default_factory=list)
    suggestion: str | None = None
    claim_ids: list[str] = Field(default_factory=list)
    heuristic: bool = False
    merged_from: list[str] = Field(default_factory=list)
    """Agent names of findings merged into this one (empty if not merged)."""


# ---------------------------------------------------------------------------
# Claim ledger
# ---------------------------------------------------------------------------


class ResearchAction(StrEnum):
    """One step type in a bounded research loop."""

    SEARCH = "search"
    FETCH_PAGE = "fetch_page"
    FIND_IN_PAGE = "find_in_page"
    ASSESS = "assess"
    STOP = "stop"


class ResearchStep(_Model):
    """One saved step of a research loop (the claim's research trail).

    ``iteration`` is 1-based and never exceeds 3 (the loop cap). ``input`` is
    the query or URL; ``summary`` is a short note on what was found. Never
    store full page text here.
    """

    agent: str
    iteration: int = Field(ge=1, le=3)
    action: ResearchAction
    input: str
    summary: str = ""
    evidence_urls: list[str] = Field(default_factory=list)
    at: datetime = Field(default_factory=utcnow)


class DebateRecord(_Model):
    """One debate round between fact-checker and devil's advocate on a claim.

    ``ruling`` is the orchestrator's decision and ``reasoning`` why.
    ``final_verdict`` is the verdict after the ruling.
    """

    factcheck_position: str
    devils_advocate_position: str
    ruling: str
    reasoning: str
    final_verdict: Verdict


class LedgerEntry(_Model):
    """Everything known about one claim.

    ``verdict_evidence`` backs ``verdict``. ``rebuttals`` are those targeting
    this claim. ``copy_edit_ids`` are IDs of copy-edit findings whose span
    overlaps the claim span. ``rechecked`` is set after the one allowed
    re-check; ``debate`` after the one allowed debate round.
    """

    claim: Claim
    verdict: Verdict = Verdict.UNCHECKED
    verdict_evidence: list[Evidence] = Field(default_factory=list)
    verdict_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    verdict_note: str = ""
    rebuttals: list[Rebuttal] = Field(default_factory=list)
    copy_edit_ids: list[str] = Field(default_factory=list)
    research_trail: list[ResearchStep] = Field(default_factory=list)
    rechecked: bool = False
    debate: DebateRecord | None = None


class AddClaim(_Model):
    """Ledger update: add a new claim (extractor)."""

    kind: Literal["add_claim"] = "add_claim"
    claim: Claim


class SetVerdict(_Model):
    """Ledger update: set a claim's fact-check verdict (fact-checker).

    Setting ``Verdict.VERIFIED`` with empty ``evidence`` is rejected.
    """

    kind: Literal["set_verdict"] = "set_verdict"
    claim_id: str
    verdict: Verdict
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    note: str = ""

    @model_validator(mode="after")
    def _verified_needs_evidence(self) -> SetVerdict:
        if self.verdict is Verdict.VERIFIED and not self.evidence:
            raise ValueError("a claim cannot be marked verified without evidence")
        return self


class AddRebuttal(_Model):
    """Ledger update: attach a rebuttal to every claim it targets."""

    kind: Literal["add_rebuttal"] = "add_rebuttal"
    rebuttal: Rebuttal


class AddResearchStep(_Model):
    """Ledger update: append a step to a claim's research trail."""

    kind: Literal["add_research_step"] = "add_research_step"
    claim_id: str
    step: ResearchStep


LedgerUpdate = Annotated[
    AddClaim | SetVerdict | AddRebuttal | AddResearchStep, Field(discriminator="kind")
]
"""Any change an agent asks the orchestrator to make to the ledger."""


class ClaimLedger(_Model):
    """The shared structured state of one review, keyed by claim ID.

    Also holds every finding produced so far (``findings``, keyed by finding
    ID) so conflicts can be detected by span overlap. Helpers raise
    ``KeyError`` for unknown claim IDs.
    """

    entries: dict[str, LedgerEntry] = Field(default_factory=dict)
    findings: dict[str, Finding] = Field(default_factory=dict)

    # -- claims ------------------------------------------------------------

    def add_claim(self, claim: Claim) -> LedgerEntry:
        """Add a claim and return its new entry. Duplicate IDs are rejected."""
        if claim.id in self.entries:
            raise ValueError(f"duplicate claim id {claim.id!r}")
        entry = LedgerEntry(claim=claim)
        self.entries[claim.id] = entry
        for finding in self.findings.values():
            if finding.agent == AgentName.COPYEDIT and _overlaps(finding.span, claim.span):
                entry.copy_edit_ids.append(finding.id)
        return entry

    def get(self, claim_id: str) -> LedgerEntry:
        """Return the entry for ``claim_id``."""
        return self.entries[claim_id]

    def claims(self, *types: ClaimType) -> list[Claim]:
        """Claims of the given types (all if none given), most important first."""
        found = [e.claim for e in self.entries.values() if not types or e.claim.type in types]
        return sorted(found, key=lambda c: (-c.importance, c.span.start))

    def thesis(self) -> Claim | None:
        """The most important thesis claim, if any."""
        theses = self.claims(ClaimType.THESIS)
        return theses[0] if theses else None

    def set_verdict(
        self,
        claim_id: str,
        verdict: Verdict,
        evidence: list[Evidence] | None = None,
        *,
        confidence: float | None = None,
        note: str = "",
    ) -> None:
        """Set a claim's verdict. Verified requires at least one evidence item."""
        update = SetVerdict(
            claim_id=claim_id,
            verdict=verdict,
            evidence=evidence or [],
            confidence=confidence,
            note=note,
        )
        entry = self.entries[claim_id]
        entry.verdict = update.verdict
        entry.verdict_evidence = list(update.evidence)
        entry.verdict_confidence = update.confidence
        entry.verdict_note = update.note

    def add_rebuttal(self, rebuttal: Rebuttal) -> None:
        """Attach a rebuttal to each claim it targets (all must exist)."""
        missing = [cid for cid in rebuttal.target_claim_ids if cid not in self.entries]
        if missing:
            raise KeyError(f"rebuttal {rebuttal.id} targets unknown claims {missing}")
        for cid in rebuttal.target_claim_ids:
            self.entries[cid].rebuttals.append(rebuttal)

    def add_research_step(self, claim_id: str, step: ResearchStep) -> None:
        """Append a research step to a claim's trail."""
        self.entries[claim_id].research_trail.append(step)

    def claims_overlapping(self, span: Span) -> list[LedgerEntry]:
        """Entries whose claim span overlaps ``span``, in document order."""
        hits = [e for e in self.entries.values() if e.claim.span.overlaps(span)]
        return sorted(hits, key=lambda e: e.claim.span.start)

    # -- findings ----------------------------------------------------------

    def add_finding(self, finding: Finding) -> None:
        """Store a finding. Copy edits are linked to overlapping claims."""
        self.findings[finding.id] = finding
        if finding.agent == AgentName.COPYEDIT and finding.span is not None:
            for entry in self.claims_overlapping(finding.span):
                if finding.id not in entry.copy_edit_ids:
                    entry.copy_edit_ids.append(finding.id)

    def remove_finding(self, finding_id: str) -> None:
        """Remove a finding and any copy-edit links to it."""
        self.findings.pop(finding_id, None)
        for entry in self.entries.values():
            if finding_id in entry.copy_edit_ids:
                entry.copy_edit_ids.remove(finding_id)

    def findings_overlapping(self, span: Span) -> list[Finding]:
        """Findings whose span overlaps ``span``, in document order."""
        hits = [f for f in self.findings.values() if _overlaps(f.span, span)]
        return sorted(hits, key=lambda f: f.span.start if f.span else 0)

    # -- updates -----------------------------------------------------------

    def apply(self, update: LedgerUpdate) -> None:
        """Apply one agent-proposed update."""
        match update:
            case AddClaim(claim=claim):
                self.add_claim(claim)
            case SetVerdict():
                self.set_verdict(
                    update.claim_id,
                    update.verdict,
                    update.evidence,
                    confidence=update.confidence,
                    note=update.note,
                )
            case AddRebuttal(rebuttal=rebuttal):
                self.add_rebuttal(rebuttal)
            case AddResearchStep(claim_id=claim_id, step=step):
                self.add_research_step(claim_id, step)

    def apply_result(self, result: AgentResult) -> None:
        """Apply every ledger update in a result, then store its findings."""
        for update in result.ledger_updates:
            self.apply(update)
        for finding in result.findings:
            self.add_finding(finding)


def _overlaps(a: Span | None, b: Span | None) -> bool:
    return a is not None and b is not None and a.overlaps(b)


# ---------------------------------------------------------------------------
# Profiles, budgets and plans
# ---------------------------------------------------------------------------


class Profile(StrEnum):
    """Review profile. ``AUTO`` asks the orchestrator to classify."""

    AUTO = "auto"
    OPINION = "opinion"
    DESIGN_DOC = "design_doc"


PROFILE_AGENTS: dict[Profile, tuple[AgentName, ...]] = {
    Profile.OPINION: (
        AgentName.EXTRACTOR,
        AgentName.FACTCHECK,
        AgentName.DEVILS_ADVOCATE,
        AgentName.COPYEDIT,
        AgentName.ORIGINALITY,
    ),
    Profile.DESIGN_DOC: (
        AgentName.EXTRACTOR,
        AgentName.FACTCHECK,
        AgentName.DEVILS_ADVOCATE,
        AgentName.STRUCTURE,
        AgentName.COPYEDIT,
    ),
}
"""Agents each concrete profile may run (from the design doc)."""

MAX_RESEARCH_ITERATIONS = 3
"""Hard cap on research-loop iterations per claim."""


class Budget(_Model):
    """Hard caps for one job (or one claim, when used per claim)."""

    max_tokens: int = Field(default=400_000, ge=0)
    max_search_calls: int = Field(default=60, ge=0)
    max_seconds: float = Field(default=600.0, ge=0)


class ClaimBudget(_Model):
    """Per-claim research budget set by the execution plan."""

    max_iterations: int = Field(default=MAX_RESEARCH_ITERATIONS, ge=1, le=MAX_RESEARCH_ITERATIONS)
    max_search_calls: int = Field(default=3, ge=0)
    max_tokens: int = Field(default=20_000, ge=0)


class ExecutionPlan(_Model):
    """The orchestrator's plan after extraction, validated against guardrails.

    ``agents`` excludes the extractor (it has already run). Claims not in
    ``claim_budgets`` get ``default_claim_budget``. ``deep_research_claim_ids``
    are claims that get the full research loop; others get a single search or
    stay unchecked when the budget runs short.
    """

    profile: Profile
    agents: list[AgentName]
    deep_research_claim_ids: list[str] = Field(default_factory=list)
    claim_budgets: dict[str, ClaimBudget] = Field(default_factory=dict)
    default_claim_budget: ClaimBudget = Field(default_factory=ClaimBudget)
    rationale: str = ""

    def budget_for(self, claim_id: str) -> ClaimBudget:
        """The research budget for one claim."""
        return self.claim_budgets.get(claim_id, self.default_claim_budget)


# ---------------------------------------------------------------------------
# Usage, results, progress and report
# ---------------------------------------------------------------------------


class Usage(_Model):
    """Resource usage. Add two with ``+``."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    search_calls: int = Field(default=0, ge=0)
    fetch_calls: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    seconds: float = Field(default=0.0, ge=0)

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            search_calls=self.search_calls + other.search_calls,
            fetch_calls=self.fetch_calls + other.fetch_calls,
            llm_calls=self.llm_calls + other.llm_calls,
            seconds=self.seconds + other.seconds,
        )


class AgentResult(_Model):
    """What one agent run returns to the orchestrator.

    ``error`` is set when the agent failed and degraded (the report then notes
    e.g. "fact-check unavailable") instead of failing the job. ``notes`` holds
    non-fatal notices (e.g. "2 quoted claims could not be located and were
    dropped"); the orchestrator may copy them into ``Report.notes``. Notes must
    never contain document text.
    """

    agent: str
    findings: list[Finding] = Field(default_factory=list)
    ledger_updates: list[LedgerUpdate] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None
    notes: list[str] = Field(default_factory=list)


class ProgressStep(StrEnum):
    """Well-known progress steps (mirrors the job lifecycle)."""

    QUEUED = "queued"
    CLASSIFYING = "classifying"
    EXTRACTING = "extracting"
    PLANNING = "planning"
    REVIEWING = "reviewing"
    REACTING = "reacting"
    RESOLVING = "resolving"
    REPORTING = "reporting"
    DONE = "done"


class ProgressEvent(_Model):
    """A user-facing progress update, e.g. "Fact-checking claim 4 of 12".

    ``step`` is usually a ``ProgressStep`` value; agents may emit their own
    sub-step messages under the current step. ``percent`` is 0–100.
    """

    step: str
    message: str
    percent: float = Field(ge=0.0, le=100.0)


class Report(_Model):
    """The final structured report, rendered to markdown and HTML.

    Sections follow the design doc order: verdict line and ``counts``;
    ``must_fix`` (factual errors, unsupported claims); ``unverified`` (factual
    claims the fact-checker could not confirm: lower severity, because failing
    to find a source is not evidence the claim is wrong); ``counter_case`` (top 3
    rebuttals); ``should_fix`` (structure); ``polish`` (copy edits);
    ``originality`` (heuristic matches); ``ledger`` (appendix). ``counts`` is
    keyed by ``Severity.label``. ``notes`` holds degradation notices such as
    "fact-check unavailable" or "partial results: time limit reached".
    """

    document_id: str
    profile: Profile
    verdict_line: str
    counts: dict[str, int] = Field(default_factory=dict)
    must_fix: list[Finding] = Field(default_factory=list)
    unverified: list[Finding] = Field(default_factory=list)
    counter_case: list[Rebuttal] = Field(default_factory=list)
    should_fix: list[Finding] = Field(default_factory=list)
    polish: list[Finding] = Field(default_factory=list)
    originality: list[Finding] = Field(default_factory=list)
    ledger: ClaimLedger = Field(default_factory=ClaimLedger)
    usage: Usage = Field(default_factory=Usage)
    notes: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)
