# Review Desk — Task Plan for Parallel Sub-Agents

Source of truth: `DESIGN.md` (exported from the Review Desk design doc). This file splits it into tasks that Claude Code sub-agents can build in parallel with minimal merge conflicts.

## How parallelism works here

The plan is built on four rules. Every sub-agent must follow them.

1. **Contracts first, then fan out.** Wave 0 (one agent, sequential) freezes the shared types, interfaces and fakes. Every later task codes against those contracts and fakes, so no task waits on another task's implementation.
2. **Each task owns directories.** A task writes only inside its owned paths (plus its own tests). If it needs a contract change, it does not edit `reviewdesk/contracts/`; it appends a request to `CONTRACT_CHANGES.md` and works around it with a local adapter.
3. **One worktree and branch per task.** Branch name `task/<id>-<slug>`. The lead session merges after review.
4. **Tests use fakes, never the network.** Every task ships unit tests that pass offline using `reviewdesk/testing/fakes.py`. Real-provider tests are marked `@pytest.mark.live` and skipped by default.

Recommended concurrency: 4–5 sub-agents at a time. More than that makes review and merging the bottleneck and burns usage quickly.

## Waves at a glance

| Wave | Tasks | Runs | Gate to next wave |
| --- | --- | --- | --- |
| 0 | T0 | 1 agent, sequential | Contracts merged, CI green |
| 1 | T1–T10 | Parallel | All merged, CI green |
| 2 | T11–T12 | Parallel (2 agents) | **Phase 0 gate:** rebuttals cite real, relevant sources on your own drafts; pipeline beats the single-prompt baseline |
| 3 | T13 then T14–T18 | T13 sequential, then parallel | Full review from Claude and ChatGPT, report arrives by email |
| 4 | T19–T21 | Parallel | Launch |

The Wave 2 gate is deliberate: hosting work (Wave 3) does not start until evidence retrieval proves itself. You can pull Wave 3 forward, but that spends effort before the core bet is validated.

---

## Wave 0 — Foundation (sequential)

### T0 · Repo scaffold and shared contracts
**Owns:** everything at repo root, `reviewdesk/contracts/`, `reviewdesk/testing/`, `.github/workflows/ci.yml`
**Depends on:** nothing

Build:
- Python 3.12 project with `uv`, `ruff`, `mypy`, `pytest`, `pytest-asyncio`. Package `reviewdesk/`.
- Directory skeleton for every owned path listed below (empty `__init__.py` files), so later tasks never create sibling folders that collide.
- `reviewdesk/contracts/models.py` (Pydantic v2):
  - `Document` (id, text, source_url, word_count)
  - `Span` (start, end), `ClaimType` (thesis | supporting | factual)
  - `Claim` (id, text, span, type, importance)
  - `Evidence` (url, title, excerpt, retrieved_at)
  - `Verdict` (verified | wrong | unsupported | unchecked)
  - `Rebuttal` (id, target_claim_ids, argument, evidence, strength 1–5)
  - `Severity` (factual_error > unsupported > strong_rebuttal > structure > style)
  - `Finding` (id, agent, severity, span, message, evidence, suggestion)
  - `LedgerEntry` (claim, verdict, verdict_evidence, rebuttals, copy_edit_ids)
  - `ClaimLedger` (entries keyed by claim id, plus helpers: `add_claim`, `set_verdict`, `add_rebuttal`, `findings_overlapping(span)`)
  - `Profile` (auto | opinion | design_doc), `Budget` (max_tokens, max_search_calls, max_seconds)
  - `AgentResult` (agent, findings, ledger_updates, usage)
  - `Report` (verdict_line, counts, must_fix, counter_case, should_fix, polish, originality, ledger, usage)
  - `ProgressEvent` (step, message, percent)
- `reviewdesk/contracts/interfaces.py` (Protocols):
  - `LLMClient.complete(messages, *, schema=None, model_tier) -> LLMResponse` (tiers: cheap | mid | strong)
  - `SearchClient.search(query, *, k) -> list[SearchResult]`
  - `Fetcher.fetch(url) -> FetchedPage`
  - `Agent.name`, `Agent.run(ctx: ReviewContext) -> AgentResult`
  - `ReviewContext` (document, profile, ledger, llm, search, fetcher, budget, emit_progress)
  - Pipeline entry point: `async def run_review(doc: Document, profile: Profile, on_progress: Callable[[ProgressEvent], None]) -> Report`
- `reviewdesk/testing/fakes.py`: `FakeLLM` (scripted responses keyed by prompt tag), `FakeSearch`, `FakeFetcher`, `FakeAgent`, sample documents and a sample populated ledger.
- CI: lint, type-check, offline tests on every push.
- `CONTRACT_CHANGES.md` (empty template) and `ARCHITECTURE.md` summarizing the contracts.

**Done when:** CI is green; every model and protocol has a docstring; fakes cover every interface.

---

## Wave 1 — Core engine (parallel)

Every Wave 1 task depends only on T0.

### T1 · LLM provider adapters
**Owns:** `reviewdesk/providers/llm/`
- Anthropic adapter first, OpenAI second, both implementing `LLMClient`.
- Tier → model mapping in config; structured output via the provider's JSON/tool mechanism, validated against the Pydantic schema.
- Retries with backoff on rate limits and timeouts; token usage returned on every call.
- Key passed in at construction; never logged.

**Done when:** offline tests with recorded fixtures pass; `live` tests pass with a real key.

### T2 · Search and fetch adapters
**Owns:** `reviewdesk/providers/search/`, `reviewdesk/providers/fetch/`, `scripts/compare_search.py`
- Two or three search API adapters behind `SearchClient` (candidates to be chosen after checking current free tiers).
- `Fetcher` with readable-text extraction, SSRF protection (block private IP ranges and non-HTTP schemes), size and redirect caps.
- `compare_search.py`: runs a fixed set of claim queries against each provider and writes a comparison table (relevance judged by hand).

**Done when:** offline tests pass; comparison script produces a table for the Phase 0 decision.

### T3 · Claim extractor agent
**Owns:** `reviewdesk/agents/extractor/`
- Splits the document into thesis, supporting claims and factual claims with exact spans and an importance score.
- Writes entries to the ledger; spans must round-trip to the original text exactly.

**Done when:** tests on 5 sample documents pass with `FakeLLM`; span integrity test passes.

### T4 · Fact-checker agent
**Owns:** `reviewdesk/agents/factcheck/`
- For each factual claim (top N by importance, within budget): generate search queries, retrieve, fetch, judge the verdict with evidence.
- Prefers primary sources; never marks a claim verified without evidence.

**Done when:** fake-backed tests cover verified, wrong, unsupported and budget-exhausted cases.

### T5 · Devil's advocate agent
**Owns:** `reviewdesk/agents/devils_advocate/`
- Takes the thesis and supporting claims, retrieves counter-evidence, and writes the strongest opposing case as `Rebuttal`s tied to claim ids.
- Rebuttals without retrieved evidence are allowed but marked for downgrade.
- Design-doc profile variant: argues against the design (alternatives, unstated assumptions).

**Done when:** fake-backed tests pass; every rebuttal references at least one claim id.

### T6 · Copy editor and structure reviewer agents
**Owns:** `reviewdesk/agents/copyedit/`, `reviewdesk/agents/structure/`
- Copy editor: line-level suggestions with spans; optional style guide input to preserve the author's voice.
- Structure reviewer: flow and missing-section findings; design-doc variant checks gaps and unstated assumptions.

**Done when:** fake-backed tests pass; all findings carry valid spans.

### T7 · Originality checker agent
**Owns:** `reviewdesk/agents/originality/`
- Samples distinctive sentences, searches for near-matches, reports possible matches with URLs.
- Every finding labelled heuristic.

**Done when:** fake-backed tests cover match and no-match cases.

### T8 · Report renderer
**Owns:** `reviewdesk/report/`
- `Report` → markdown (inline MCP return) and HTML email with plain-text fallback, following the section order in the design doc.
- Snapshot tests for both formats.

**Done when:** snapshot tests pass for a sample report with every section populated and with empty sections.

### T9 · Evaluation set and harness
**Owns:** `eval/`
- Harness that runs any `run_review`-shaped callable over a dataset and computes the metrics in the design doc: recall per defect type, precision, citation validity, rebuttal strength (LLM judge), conflict-resolution accuracy, cost and latency.
- Baseline runner: a single-prompt reviewer using `LLMClient`.
- Seed dataset format with answer keys, plus the first 10 seeded documents (grow to 30 in Wave 2).

**Done when:** harness runs end to end against `FakeAgent`-based pipelines and prints a results table.

### T10 · Orchestrator
**Owns:** `reviewdesk/orchestrator/`, `reviewdesk/pipeline.py`
- Profile classification (cheap tier), execution plan (extractor first, then the rest in parallel with `asyncio.gather`), budgets, progress events.
- Conflict resolution as pure functions over the ledger, implementing every rule in the design doc; ranking by severity.
- `run_review` implementation composing agents supplied by a registry, so it runs fully on `FakeAgent`s.

**Done when:** table-driven tests cover every conflict rule; a full fake pipeline produces a valid `Report` and a correct progress sequence.

---

## Wave 2 — Integration and the Phase 0 gate (parallel)

### T11 · Wire the real pipeline and CLI
**Owns:** `reviewdesk/cli/`, `reviewdesk/registry.py`
**Depends on:** T1–T8, T10
- Registry wiring real agents and providers; `reviewdesk review <file-or-url> [--profile]` prints the markdown report and progress.
- Key and search config from environment variables.

**Done when:** a real review of one of your drafts completes end to end.

### T12 · Local MCP server (stdio)
**Owns:** `reviewdesk/mcp_local/`
**Depends on:** T10 (contract only; build against the fake pipeline, then switch to the real one)
- One tool, `review_document` (content or url, profile, focus), over stdio using the official MCP Python SDK.
- Streams `notifications/progress` from `ProgressEvent`s; returns the markdown report.
- Clear errors: missing key, document too long, unreachable URL.

**Done when:** works from Claude Desktop and Claude Code on a real draft.

**Then run the Phase 0 gate** (you, not an agent): run T9's harness with the real pipeline and the baseline; check rebuttal sources on your drafts. Grow the dataset to 30 documents.

---

## Wave 3 — Hosted service

### T13 · Hosted scaffold (sequential, runs first)
**Owns:** `reviewdesk/db/`, `migrations/`, `reviewdesk/contracts/hosted.py`
- Postgres schema from the design doc (`users`, `provider_keys`, `jobs`) with migrations.
- Repository interfaces plus in-memory fakes; `JobStatus` state machine; `KeyVault` and `Mailer` protocols.

**Done when:** migrations apply to a fresh database; fakes cover every repository.

The following run in parallel after T13.

### T14 · Google sign-in and MCP OAuth
**Owns:** `reviewdesk/web/auth/`
- The server acts as the OAuth 2.1 authorization server toward MCP clients and delegates login to Google.
- Issues its own short-lived access tokens and hashed refresh tokens; never passes Google's tokens to clients.

**Done when:** the OAuth flow completes from an MCP client against a local deployment.

### T15 · Setup page and key vault
**Owns:** `reviewdesk/web/setup/`, `reviewdesk/security/`
- Setup page: choose provider, paste key, validate with a cheap test call; rotate, delete key, delete account.
- Envelope encryption: per-user data key wrapped by a KMS or secrets-manager master key; decrypt only inside workers.

**Done when:** keys round-trip encrypted; no plaintext key appears in logs or the database (a test asserts this).

### T16 · Job queue and workers
**Owns:** `reviewdesk/jobs/`
- Durable queue, workers running `run_review`, progress written to the job record, retries, per-job timeout with partial results, idempotency by content hash, purge of document and report after delivery.

**Done when:** a job runs end to end on fakes, including timeout, failure and duplicate-submit paths.

### T17 · Email delivery
**Owns:** `reviewdesk/delivery/`
- `Mailer` adapter for a transactional email provider; report email using T8's HTML; failure email.
- Sends only to the account's verified address.

**Done when:** tests on a fake mailer pass; a live send reaches your inbox.

### T18 · Remote MCP server
**Owns:** `reviewdesk/mcp_remote/`
- Streamable HTTP transport with OAuth from T14.
- Response-mode selection from client capabilities: MCP Tasks, progress notifications, or fire-and-forget; relays progress from the job record.

**Done when:** a review works from Claude and ChatGPT, with the report emailed in every mode.

---

## Wave 4 — Harden and launch (parallel)

### T19 · Infrastructure and deployment
**Owns:** `infra/`, `Dockerfile`, deploy config
- Container build, managed Postgres, secrets, scale-to-zero hosting, SPF and DKIM for the email domain.

### T20 · Observability and rate limits
**Owns:** `reviewdesk/observability/`, `reviewdesk/ratelimit/`
- OpenTelemetry traces per job and per agent call, with content redacted; per-user daily cap, per-job budgets, global circuit breaker on search.

### T21 · Eval in CI and README
**Owns:** `.github/workflows/eval.yml`, `README.md`, `docs/`
- Eval subset in CI on changes to agents, prompts or the orchestrator; README with the results table against the baseline and the privacy promise.

---

## Lead-session kickoff prompt (Claude Code)

Paste this into the main Claude Code session in the repo:

```
Read TASKS.md and DESIGN.md. You are the lead. Run Wave 0 (T0) yourself.
When T0 is merged and CI is green, launch Wave 1 using the task-implementer
sub-agent, one task per sub-agent, each in its own worktree, at most 5 at a time.
Give each sub-agent: its task section verbatim, its owned paths, and the rule
that it must not edit files outside them. When a sub-agent finishes, review its
diff against the task's "Done when", run the full test suite, and merge. Collect
any CONTRACT_CHANGES.md requests and resolve them before launching the next
batch. Stop after Wave 2 and report results so I can run the Phase 0 gate.
```
