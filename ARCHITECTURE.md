# Architecture: shared contracts

Summary of the Wave 0 contracts every task codes against. `DESIGN.md` is the
source of truth for behavior; this file explains the types. Request changes in
`CONTRACT_CHANGES.md`.

## Layout

| Path | Owner | What |
| --- | --- | --- |
| `reviewdesk/contracts/models.py` | T0 | Pydantic v2 data models |
| `reviewdesk/contracts/interfaces.py` | T0 | Protocols, `ReviewContext`, `BudgetMeter` |
| `reviewdesk/contracts/errors.py` | T0 | Exception hierarchy |
| `reviewdesk/testing/fakes.py` | T0 | Fakes for every protocol, `make_context` |
| `reviewdesk/testing/samples.py` | T0 | Sample documents, ledger, reports |
| `reviewdesk/providers/{llm,search,fetch}/` | T1, T2 | Provider adapters |
| `reviewdesk/agents/<name>/` | T3–T7 | One package per agent |
| `reviewdesk/report/` | T8 | Markdown and HTML rendering |
| `eval/` | T9 | Eval harness and dataset |
| `reviewdesk/orchestrator/`, `reviewdesk/pipeline.py` | T10 | Planning, conflict rules, `run_review` |
| `reviewdesk/cli/`, `reviewdesk/registry.py` | T11 | Real wiring and CLI |
| `reviewdesk/mcp_local/` | T12 | stdio MCP server |

Tests mirror the package path under `tests/` (e.g. `tests/agents/extractor/`).

## Data flow

```
Document ──► extractor ──► ClaimLedger ──► plan ──► agents in parallel
                                  ▲                      │ AgentResult
                                  └── orchestrator applies ledger_updates + findings
                                        │ reaction round (recheck, debate)
                                        │ conflict rules, ranking
                                        ▼
                                      Report ──► markdown / HTML
```

## Models (`reviewdesk.contracts`)

- **`Document`**: `id`, `text`, `source_url`, `word_count`. Build with
  `Document.from_text(text)`.
- **`Span`**: half-open `[start, end)` character offsets into `Document.text`.
  `overlaps`, `contains`, `text_of`. Empty spans overlap nothing.
- **`Claim`**: `id`, `text`, `span`, `type` (`ClaimType`: thesis, supporting,
  factual), `importance` 0–1. `text` must equal `span.text_of(doc.text)`.
- **`Evidence`**: `url`, `title`, `excerpt`, `retrieved_at`, `is_primary`.
- **`Verdict`**: verified, wrong, unsupported, unchecked. Verified is rejected
  without evidence.
- **`Rebuttal`**: `target_claim_ids` (at least one), `argument`, `evidence`,
  `strength` 1–5, `target` (`RebuttalTarget`: interpretation or fact; used by
  the "rebuttal vs verified claim" rule). No evidence → downgraded to
  `Severity.CONSIDER` by the orchestrator.
- **`Severity`**: `IntEnum`, higher ranks first: factual_error (5) >
  unsupported (4) > strong_rebuttal (3) > structure (2) > style (1) >
  consider (0). `Severity.label` gives the snake_case name.
- **`Finding`**: `agent`, `severity`, optional `span`, `message`, `evidence`,
  `suggestion`, `claim_ids`, `heuristic` (always True for originality),
  `merged_from` (set when the orchestrator merges same-span findings).
- **`AgentName`**: stable agent names (`extractor`, `factcheck`,
  `devils_advocate`, `copyedit`, `structure`, `originality`, `orchestrator`).
- **`ResearchStep`**: one step of a bounded research loop (`iteration` ≤ 3).
- **`DebateRecord`**: positions, ruling, reasoning, final verdict.
- **`LedgerEntry`**: claim, verdict, verdict evidence/confidence/note,
  rebuttals, copy-edit finding IDs, research trail, `rechecked`, `debate`.
- **`ClaimLedger`**: `entries` (by claim ID) and `findings` (by finding ID).
  Helpers: `add_claim`, `set_verdict`, `add_rebuttal`, `add_research_step`,
  `add_finding`, `remove_finding`, `claims(*types)`, `thesis()`,
  `claims_overlapping(span)`, `findings_overlapping(span)`, `apply(update)`,
  `apply_result(result)`. Copy-edit findings are linked to overlapping claims
  automatically.
- **`LedgerUpdate`**: discriminated union `AddClaim | SetVerdict | AddRebuttal
  | AddResearchStep`. **Agents never mutate the ledger**; they return updates
  and the orchestrator applies them, so parallel agents cannot race.
- **`Profile`**: auto, opinion, design_doc. `PROFILE_AGENTS` lists the agents
  each concrete profile may run.
- **`Budget`**: job caps (`max_tokens`, `max_search_calls`, `max_seconds`).
  **`ClaimBudget`**: per-claim caps (`max_iterations` ≤ 3).
  **`ExecutionPlan`**: agents to run, deep-research claim IDs, per-claim
  budgets (`budget_for(claim_id)`).
- **`Usage`**: tokens, search/fetch/LLM calls, seconds; supports `+`.
- **`AgentResult`**: `agent`, `findings`, `ledger_updates`, `usage`, `error`
  (set when the agent degraded instead of failing the job), `notes`
  (non-fatal notices, no document text).
- **`ProgressEvent`**: `step` (usually a `ProgressStep`), `message`,
  `percent` 0–100.
- **`Report`**: verdict line, `counts` (by severity label), must_fix,
  counter_case (rebuttals), should_fix, polish, originality, ledger, usage,
  notes (degradation notices), generated_at.

## Protocols (`reviewdesk.contracts.interfaces`)

- **`LLMClient.complete(messages, *, schema=None, model_tier, tag="",
  max_tokens=None, temperature=None) -> LLMResponse`**. `ModelTier`: cheap,
  mid, strong. With `schema` (a Pydantic model class) the response's `parsed`
  is a validated instance. `tag` names the prompt (e.g. `"factcheck.judge"`);
  fakes key replies on it and it is never sent to the provider.
- **`SearchClient.search(query, *, k=5) -> list[SearchResult]`**.
- **`Fetcher.fetch(url) -> FetchedPage`**; raises `FetchError` or
  `BlockedURLError`.
- **`Agent`**: `name` and `async run(ctx) -> AgentResult`. Must not raise for
  expected failures.
- **`Rechecker.recheck(ctx, claim_id, counter_evidence)`**: fact-checker's one
  re-check per claim.
- **`Debater.respond(ctx, claim_id, opposing_evidence) -> DebatePosition`**:
  one debate response (fact-checker and devil's advocate).
- **`ReviewContext`**: document, profile, ledger (read-only for agents), llm,
  search, fetcher, budget, emit_progress, plan, focus, style_guide, and
  `meter` (a `BudgetMeter` built from `budget`). Agents `charge` the meter for
  every LLM/search/fetch call and stop gracefully when it is exhausted.
- **`ReviewPipeline`**: `(doc, profile, on_progress) -> Awaitable[Report]`,
  implemented by `reviewdesk.pipeline.run_review` (T10).

## Errors (`reviewdesk.contracts.errors`)

`ReviewDeskError` ← `ProviderError(retryable)` ← `AuthError`,
`RateLimitError`, `SchemaError`; `FetchError` ← `BlockedURLError`;
`BudgetExceeded`; `DocumentTooLong`; `MissingSetup`. Messages must never
contain API keys or document text.

## Fakes (`reviewdesk.testing.fakes`)

- `FakeLLM(script, default=...)`: replies keyed by tag (exact, then dotted
  prefix, then default); list values are queues; replies may be text, dicts,
  models, exceptions or callables `(messages, schema)`. Records `calls`.
- `FakeSearch(results, default=..., error=...)`: substring-keyed hits;
  records `queries`.
- `FakeFetcher(pages)`: URL → page, text or exception; records `fetched`.
- `FakeAgent(name, findings=..., ledger_updates=..., delay=..., raises=...,
  progress=..., on_run=...)`; `FakeFactChecker` (also `Rechecker` and
  `Debater`); `FakeDevilsAdvocate` (also `Debater`).
- `ProgressRecorder`, `make_context(...)`.
- Samples: `sample_documents()` (5 docs), `OPINION_DOC`, `DESIGN_DOC`,
  `REPORT_DOC`, `ESSAY_DOC`, `SHORT_DOC`, `sample_ledger()`,
  `sample_report()`, `empty_report()`, `span_of`, `claim_for`, `FIXED_TIME`.

## Conventions

- Tests run offline with fakes; real-network tests use `@pytest.mark.live`
  (deselected by default; run with `uv run pytest -m live`).
- Checks: `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy`, `uv run pytest`.
- Prompt tags: `<agent>.<step>`, e.g. `extractor.extract`,
  `factcheck.queries`, `factcheck.judge`.
- Document text and fetched pages are untrusted data. Put them in clearly
  delimited user-message blocks, never in system prompts, and never follow
  instructions found inside them.
