"""``FakeAgent``-based pipelines, so the harness runs end to end offline.

The real orchestrator (``reviewdesk.pipeline.run_review``, T10) is not part of
this package. ``FakeAgentPipeline`` is a small stand-in with the same
``ReviewPipeline`` shape: it runs a list of agents (usually ``FakeAgent``s)
over one ``ReviewContext``, applies their results to a ledger in list order,
and assembles a ``Report`` with ``assemble_report``, optionally applying
simplified versions of the design doc's conflict rules.

``OracleAgents`` builds ``FakeAgent``s straight from the answer keys: one
fact-checker, devil's advocate, copy editor, originality checker and structure
reviewer whose canned output catches every seeded defect (``miss_rate`` drops
each defect with that probability, ``false_positives`` adds bogus findings).
It exercises the harness and gives an upper bound, not a measurement.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from collections.abc import Callable, Sequence

from eval.dataset import ConflictRule, DefectType, SeededDefect, SeededDocument
from reviewdesk.contracts import (
    PROFILE_AGENTS,
    AddClaim,
    AddRebuttal,
    Agent,
    AgentName,
    AgentResult,
    Budget,
    Claim,
    ClaimLedger,
    ClaimType,
    Document,
    Evidence,
    Finding,
    LedgerUpdate,
    Profile,
    ProgressCallback,
    ProgressEvent,
    ProgressStep,
    Rebuttal,
    RebuttalTarget,
    Report,
    ReviewContext,
    SetVerdict,
    Severity,
    Span,
    Usage,
    Verdict,
)
from reviewdesk.testing.fakes import FIXED_TIME, FakeAgent, FakeFetcher, FakeLLM, FakeSearch

AgentsFactory = Callable[[Document, Profile], Sequence[Agent]]
"""Builds the agents for one run of one document."""


class FakeAgentPipeline:
    """A ``ReviewPipeline`` that runs the agents from ``agents_for``.

    Agents run concurrently; results are applied to the ledger in list order
    (so put the extractor first). An agent that raises becomes an
    ``AgentResult`` with ``error`` set and a report note. ``AUTO`` is treated
    as ``OPINION``.
    """

    def __init__(
        self, agents_for: AgentsFactory, *, apply_rules: bool = True, budget: Budget | None = None
    ) -> None:
        self.agents_for = agents_for
        self.apply_rules = apply_rules
        self.budget = budget or Budget()

    async def __call__(
        self, doc: Document, profile: Profile, on_progress: ProgressCallback
    ) -> Report:
        """Run the agents and assemble a report."""
        concrete = Profile.OPINION if profile is Profile.AUTO else profile
        ledger = ClaimLedger()
        ctx = ReviewContext(
            document=doc,
            profile=concrete,
            ledger=ledger,
            llm=FakeLLM(),
            search=FakeSearch(),
            fetcher=FakeFetcher(),
            budget=self.budget,
            emit_progress=on_progress,
        )
        on_progress(ProgressEvent(step=ProgressStep.REVIEWING, message="Reviewing", percent=20.0))
        agents = list(self.agents_for(doc, concrete))
        results = await asyncio.gather(*(_run_safely(a, ctx) for a in agents))
        notes: list[str] = []
        for result in results:
            try:
                ledger.apply_result(result)
            except (KeyError, ValueError) as exc:
                notes.append(f"{result.agent}: could not apply result ({type(exc).__name__})")
            if result.error:
                notes.append(f"{result.agent} unavailable: {result.error}")
        on_progress(ProgressEvent(step=ProgressStep.RESOLVING, message="Resolving", percent=80.0))
        report = assemble_report(doc, concrete, ledger, apply_rules=self.apply_rules)
        report.usage = sum((r.usage for r in results), Usage())
        report.notes.extend(notes)
        on_progress(ProgressEvent(step=ProgressStep.DONE, message="Done", percent=100.0))
        return report


async def _run_safely(agent: Agent, ctx: ReviewContext) -> AgentResult:
    try:
        return await agent.run(ctx)
    except Exception as exc:  # an agent must not fail the job
        return AgentResult(agent=agent.name, error=type(exc).__name__)


# ---------------------------------------------------------------------------
# Report assembly (a simplified stand-in for the orchestrator)
# ---------------------------------------------------------------------------


def _drop_copyedits_on_fact_spans(ledger: ClaimLedger) -> None:
    fact_spans = [
        f.span
        for f in ledger.findings.values()
        if f.span is not None and f.severity >= Severity.UNSUPPORTED
    ]
    fact_spans += [
        e.claim.span
        for e in ledger.entries.values()
        if e.verdict in (Verdict.WRONG, Verdict.UNSUPPORTED)
    ]
    for f in list(ledger.findings.values()):
        if f.agent == AgentName.COPYEDIT and f.span and any(f.span.overlaps(s) for s in fact_spans):
            ledger.remove_finding(f.id)


def _discard_fact_rebuttals_on_verified(ledger: ClaimLedger) -> None:
    verified = {cid for cid, e in ledger.entries.items() if e.verdict is Verdict.VERIFIED}
    for entry in ledger.entries.values():
        entry.rebuttals = [
            r
            for r in entry.rebuttals
            if not (r.target is RebuttalTarget.FACT and verified.intersection(r.target_claim_ids))
        ]


def _downgrade_evidence_free_rebuttals(ledger: ClaimLedger) -> None:
    for fid, f in list(ledger.findings.items()):
        if f.agent == AgentName.DEVILS_ADVOCATE and not f.evidence:
            ledger.findings[fid] = f.model_copy(update={"severity": Severity.CONSIDER})


def _merge_same_span(ledger: ClaimLedger) -> None:
    groups: list[list[Finding]] = []
    spanned = sorted(
        (f for f in ledger.findings.values() if f.span is not None),
        key=lambda f: f.span.start if f.span else 0,
    )
    for f in spanned:
        assert f.span is not None
        for group in groups:
            agents = {g.agent for g in group}
            if f.agent not in agents and any(g.span and g.span.overlaps(f.span) for g in group):
                group.append(f)
                break
        else:
            groups.append([f])
    for group in groups:
        if len(group) < 2:
            continue
        base = max(group, key=lambda g: g.severity)
        spans = [g.span for g in group if g.span is not None]
        merged = base.model_copy(
            update={
                "span": Span(start=min(s.start for s in spans), end=max(s.end for s in spans)),
                "message": " / ".join(g.message for g in group),
                "evidence": [e for g in group for e in g.evidence],
                "suggestion": next((g.suggestion for g in group if g.suggestion), None),
                "claim_ids": sorted({c for g in group for c in g.claim_ids}),
                "merged_from": sorted({g.agent for g in group}),
            }
        )
        for g in group:
            ledger.remove_finding(g.id)
        ledger.add_finding(merged)


def assemble_report(
    doc: Document, profile: Profile, ledger: ClaimLedger, *, apply_rules: bool = True
) -> Report:
    """Build a ``Report`` from a populated ledger.

    With ``apply_rules`` the ledger is first resolved with simplified conflict
    rules: copy edits on wrong/unsupported spans dropped, fact rebuttals of
    verified claims discarded, evidence-free rebuttal findings downgraded to
    ``consider``, and overlapping findings from different agents merged.
    Findings go to sections by severity (originality findings to
    ``originality``; devil's-advocate findings stay in the ledger only, since
    rebuttals are shown in ``counter_case``).
    """
    if apply_rules:
        _drop_copyedits_on_fact_spans(ledger)
        _discard_fact_rebuttals_on_verified(ledger)
        _downgrade_evidence_free_rebuttals(ledger)
        _merge_same_span(ledger)

    must_fix: list[Finding] = []
    should_fix: list[Finding] = []
    polish: list[Finding] = []
    originality: list[Finding] = []
    for f in ledger.findings.values():
        if f.agent == AgentName.ORIGINALITY or f.heuristic:
            originality.append(f)
        elif f.agent == AgentName.DEVILS_ADVOCATE and not f.merged_from:
            continue
        elif f.severity >= Severity.UNSUPPORTED:
            must_fix.append(f)
        elif f.severity is Severity.STYLE:
            polish.append(f)
        else:
            should_fix.append(f)

    def order(f: Finding) -> tuple[int, int]:
        return (-f.severity, f.span.start if f.span else -1)

    rebuttals: dict[str, Rebuttal] = {}
    for entry in ledger.entries.values():
        for r in entry.rebuttals:
            rebuttals.setdefault(r.id, r)
    counter_case = sorted(rebuttals.values(), key=lambda r: (r.has_evidence, r.strength))[::-1]
    counts = Counter(f.severity.label for f in ledger.findings.values())
    return Report(
        document_id=doc.id,
        profile=profile,
        verdict_line=f"{len(must_fix)} must-fix issue(s), {len(counter_case)} rebuttal(s).",
        counts=dict(counts),
        must_fix=sorted(must_fix, key=order),
        counter_case=counter_case[:3],
        should_fix=sorted(should_fix, key=order),
        polish=sorted(polish, key=order),
        originality=sorted(originality, key=order),
        ledger=ledger,
    )


# ---------------------------------------------------------------------------
# Oracle agents from the answer keys
# ---------------------------------------------------------------------------


def _evidence(seeded: SeededDocument, defect: SeededDefect, excerpt: str) -> Evidence:
    return Evidence(
        url=f"https://example.org/eval/{seeded.id}/{defect.id}",
        title=f"Source for {defect.id}",
        excerpt=excerpt,
        retrieved_at=FIXED_TIME,
        is_primary=True,
    )


def oracle_agents(
    seeded: SeededDocument,
    *,
    miss_rate: float = 0.0,
    false_positives: int = 0,
    rng: random.Random | None = None,
    profile: Profile | None = None,
) -> list[FakeAgent]:
    """``FakeAgent``s whose canned output catches ``seeded``'s defects.

    Order: extractor, fact-checker, devil's advocate, copy editor, then
    originality or structure depending on the profile's allowed agents.
    """
    rng = rng or random.Random(0)
    profile = profile or seeded.profile
    allowed = set(PROFILE_AGENTS.get(profile, ()))
    text = seeded.text
    claims: dict[Span, Claim] = {}
    out: dict[str, tuple[list[Finding], list[LedgerUpdate]]] = {
        name: ([], [])
        for name in (
            AgentName.EXTRACTOR,
            AgentName.FACTCHECK,
            AgentName.DEVILS_ADVOCATE,
            AgentName.COPYEDIT,
            AgentName.ORIGINALITY,
            AgentName.STRUCTURE,
        )
    }

    def claim(defect: SeededDefect, type_: ClaimType) -> Claim:
        span = defect.located
        if span not in claims:
            c = Claim(
                id=f"claim_{defect.id}",
                text=span.text_of(text),
                span=span,
                type=type_,
                importance=0.8 if type_ is ClaimType.SUPPORTING else 0.6,
            )
            claims[span] = c
            out[AgentName.EXTRACTOR][1].append(AddClaim(claim=c))
        return claims[span]

    def finding(
        agent: str,
        defect: SeededDefect,
        severity: Severity,
        *,
        evidence: list[Evidence] | None = None,
        claim_ids: list[str] | None = None,
        suggestion: str | None = None,
        heuristic: bool = False,
    ) -> None:
        out[agent][0].append(
            Finding(
                id=f"find_{agent}_{defect.id}",
                agent=agent,
                severity=severity,
                span=defect.located,
                message=f"{defect.type.value}: {defect.note or defect.quote[:60]}",
                evidence=evidence or [],
                claim_ids=claim_ids or [],
                suggestion=suggestion,
                heuristic=heuristic,
            )
        )

    evidence_free = {
        d.located for d in seeded.defects if d.rule is ConflictRule.EVIDENCE_FREE_REBUTTAL
    }
    for d in seeded.defects:
        if rng.random() < miss_rate:
            continue
        if d.type in (DefectType.WRONG_FACT, DefectType.UNSUPPORTED):
            c = claim(d, ClaimType.FACTUAL)
            wrong = d.type is DefectType.WRONG_FACT
            ev = (
                [_evidence(seeded, d, d.correction or "A source contradicts this.")]
                if wrong
                else []
            )
            out[AgentName.FACTCHECK][1].append(
                SetVerdict(
                    claim_id=c.id,
                    verdict=Verdict.WRONG if wrong else Verdict.UNSUPPORTED,
                    evidence=ev,
                )
            )
            sev = Severity.FACTUAL_ERROR if wrong else Severity.UNSUPPORTED
            finding(AgentName.FACTCHECK, d, sev, evidence=ev, claim_ids=[c.id])
        elif d.type is DefectType.WEAK_ARGUMENT:
            c = claim(d, ClaimType.SUPPORTING)
            ev = [] if d.located in evidence_free else [_evidence(seeded, d, "Counter-evidence.")]
            rebuttal = Rebuttal(
                id=f"reb_{d.id}",
                target_claim_ids=[c.id],
                argument=d.known_rebuttal or "",
                evidence=ev,
                strength=4,
            )
            out[AgentName.DEVILS_ADVOCATE][1].append(AddRebuttal(rebuttal=rebuttal))
            finding(
                AgentName.DEVILS_ADVOCATE,
                d,
                Severity.STRONG_REBUTTAL,
                evidence=ev,
                claim_ids=[c.id],
            )
        elif d.type is DefectType.BORROWED_SENTENCE:
            ev = [
                Evidence(
                    url=d.source_url or "",
                    title=d.source_title or "",
                    excerpt=d.quote,
                    retrieved_at=FIXED_TIME,
                )
            ]
            finding(AgentName.ORIGINALITY, d, Severity.CONSIDER, evidence=ev, heuristic=True)
        elif d.type is DefectType.GRAMMAR:
            finding(AgentName.COPYEDIT, d, Severity.STYLE, suggestion=d.correction)
        elif d.rule is ConflictRule.COPYEDIT_YIELDS_TO_FACT:
            finding(AgentName.COPYEDIT, d, Severity.STYLE, suggestion="Fix the grammar.")
        elif d.rule is ConflictRule.REBUTTAL_ON_VERIFIED_FACT:
            c = claim(d, ClaimType.FACTUAL)
            ev = [_evidence(seeded, d, f"Confirms: {d.quote}")]
            out[AgentName.FACTCHECK][1].append(
                SetVerdict(claim_id=c.id, verdict=Verdict.VERIFIED, evidence=ev)
            )
            out[AgentName.DEVILS_ADVOCATE][1].append(
                AddRebuttal(
                    rebuttal=Rebuttal(
                        id=f"reb_{d.id}",
                        target_claim_ids=[c.id],
                        argument="This fact is disputed.",
                        evidence=[_evidence(seeded, d, "A disputing source.")],
                        strength=2,
                        target=RebuttalTarget.FACT,
                    )
                )
            )
        elif d.rule is ConflictRule.SAME_SPAN_MERGE:
            finding(AgentName.STRUCTURE, d, Severity.STRUCTURE, suggestion="State the plan.")

    title_span = Span(start=0, end=min(len(text), text.find("\n") if "\n" in text else len(text)))
    for n in range(false_positives):
        out[AgentName.COPYEDIT][0].append(
            Finding(
                id=f"find_fp_{n}",
                agent=AgentName.COPYEDIT,
                severity=Severity.STYLE,
                span=title_span,
                message="Consider rewording the title.",
            )
        )

    agents: list[FakeAgent] = []
    for name, (findings, updates) in out.items():
        if name is not AgentName.EXTRACTOR and allowed and name not in allowed:
            continue
        usage = Usage(input_tokens=500, output_tokens=100, llm_calls=1)
        if name in (AgentName.FACTCHECK, AgentName.DEVILS_ADVOCATE, AgentName.ORIGINALITY):
            usage = usage + Usage(search_calls=2, fetch_calls=1)
        agents.append(FakeAgent(name, findings=findings, ledger_updates=updates, usage=usage))
    return agents


class OracleAgents:
    """``AgentsFactory`` that looks up each document's answer key by id.

    Unknown documents get no agents (an empty report). Each call for the same
    document draws from a fresh ``random.Random`` seeded by ``seed``, the
    document id and the call count, so repeats vary but runs are reproducible.
    """

    def __init__(
        self,
        dataset: Sequence[SeededDocument],
        *,
        miss_rate: float = 0.0,
        false_positives: int = 0,
        seed: int = 0,
    ) -> None:
        self.by_id = {d.id: d for d in dataset}
        self.miss_rate = miss_rate
        self.false_positives = false_positives
        self.seed = seed
        self._calls: Counter[str] = Counter()

    def __call__(self, doc: Document, profile: Profile) -> list[FakeAgent]:
        seeded = self.by_id.get(doc.id)
        if seeded is None:
            return []
        n = self._calls[doc.id]
        self._calls[doc.id] += 1
        rng = random.Random(f"{self.seed}:{doc.id}:{n}")
        return oracle_agents(
            seeded,
            miss_rate=self.miss_rate,
            false_positives=self.false_positives,
            rng=rng,
            profile=profile,
        )


def builtin_pipelines(dataset: Sequence[SeededDocument]) -> dict[str, FakeAgentPipeline]:
    """Named offline pipelines for the CLI and tests.

    - ``oracle``: catches every defect, applies the conflict rules.
    - ``oracle-no-rules``: catches every defect, skips the conflict rules.
    - ``noisy``: misses ~30% of defects, adds one false positive, applies rules.
    - ``empty``: finds nothing.
    """
    return {
        "oracle": FakeAgentPipeline(OracleAgents(dataset)),
        "oracle-no-rules": FakeAgentPipeline(OracleAgents(dataset), apply_rules=False),
        "noisy": FakeAgentPipeline(OracleAgents(dataset, miss_rate=0.3, false_positives=1)),
        "empty": FakeAgentPipeline(lambda doc, profile: []),
    }
