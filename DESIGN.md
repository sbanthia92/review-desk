# Review Desk — Design Doc

Exported 2026-09-30 from the Claude Docs design doc. The doc is the source of truth; update this file if the doc changes.

## Overview

Review Desk is a hosted, multi-agent document reviewer exposed as a single MCP tool, `review_document`. A user connects it to Claude, ChatGPT or Cursor, signs in with Google once, adds their own model API key, and asks for a review of any draft or URL. An orchestrator dispatches specialist agents in parallel, resolves their disagreements, and emails one ranked report.

**Problem.** Existing AI critique tools argue from the model's own knowledge. They fix grammar or generate generic objections, but they do not check facts against sources or build an opposing case from real evidence.

**Differentiator.** An evidence-backed review: every claim is extracted and verified with citations, and a devil's advocate builds the strongest counter-argument from retrieved sources.

**Goals**

- Review any prose document (opinion piece, technical design doc, report) through one MCP tool.
- Ground fact-checks and rebuttals in cited sources, not model memory.
- Keep all orchestration server-side and hidden behind `review_document`.
- Handle long reviews: async with email delivery, plus live progress where the client supports it.
- Retain nothing beyond what one-time setup needs; documents and reports are deleted after delivery.
- Measure quality with a seeded-error evaluation set.
- Serve as the review pipeline for the author's own football blog.

**Working name:** Review Desk (placeholder).

## Non-goals

- **Writing for the user.** Agents critique and verify; they never draft or rewrite the argument.
- **Granular public tools.** No `extract_claims` or `find_counterevidence` tools; exposing them would hand orchestration to the client.
- **Hosted models.** The service never pays for inference; users bring their own key.
- **Guaranteed plagiarism detection.** Originality checks are a heuristic and are labelled as such.
- **Auto-publishing or social posting.** Out of scope for v1.
- **Monetization.** None; this is a free portfolio and personal project.

## User experience

Setup takes about a minute, once; every review after that is one sentence in chat.

**One-time setup (web)**

1. User adds the Review Desk MCP server URL in their client (Claude, ChatGPT, Cursor).
2. On first use the client runs the OAuth flow; the user clicks **Sign in with Google**. Google supplies a verified email, so no separate email verification is needed.
3. A setup page asks for their model provider and API key. The key is validated with a cheap test call, then encrypted and stored.

**Each review (chat)**

1. User: "Review this draft" (pasted text, or a URL).
2. The client calls `review_document`.
3. The server enqueues a job and responds in one of three ways, based on client capability:
   1. **MCP Tasks supported:** returns a task handle; the client shows live status.
   2. **Progress notifications supported:** holds the call open and streams steps ("Fact-checking claim 4 of 12").
   3. **Neither:** returns immediately: "Review started; the report will be emailed to you."
4. The report is always emailed. When the call is synchronous, it is also returned inline.
5. The document and report are deleted after delivery.

## System architecture

The MCP server stays thin: it authenticates, enqueues, and relays status. All agent work happens in workers.

```
MCP clients (Claude, ChatGPT, Cursor)
        │ MCP
        ▼
MCP + setup server (OAuth, review_document) ──OAuth──► Google Sign-in
        │ enqueue                 │ keys, job status
        ▼                         ▼
     Job queue                 Postgres (users, keys, job status)
        │ jobs                    ▲ progress
        ▼                         │
Workers: orchestrator + agents ───┘      (decrypt key via KMS)
  Orchestrator · Claim extractor · Fact-checker · Devil's advocate
  · Copy editor · Originality · Structure — all share one claim ledger
        │ inference        │ search                 │ send report
        ▼                  ▼                        ▼
 Model provider      Search API + page fetch    Email provider
 (user's own key)    (evidence, citations)      (verified email)
```

Workers decrypt the user's key through KMS for the job's duration, call the model provider and search, write progress to Postgres for the MCP server to relay, and email the final report.

## Agents and orchestration

The orchestrator's conflict resolution is the core of the system; most design effort goes there.

| Agent | Job | Input | Output | Model tier |
| --- | --- | --- | --- | --- |
| Orchestrator | Classify document, pick profile, dispatch agents, resolve conflicts, rank findings | Document + profile | Final report | Strong |
| Claim extractor | Split into thesis, supporting claims, factual claims (numbers, dates, attributions) | Document | Claim ledger entries | Cheap |
| Fact-checker | Verify each factual claim against retrieved sources | Claim ledger | Verdict per claim: verified / wrong / unsupported, with citations | Mid |
| Devil's advocate | Retrieve counter-evidence and build the strongest opposing case | Thesis + supporting claims | Rebuttals, each tied to claims and sources; exposure rating | Strong |
| Copy editor | Grammar, clarity, concision; preserves author voice | Document + optional style guide | Line-level suggestions | Cheap |
| Originality checker | Sample distinctive sentences, search for near-matches | Document | Possible matches with URLs (heuristic) | Cheap |
| Structure reviewer | Flow, missing sections, argument order (design docs: gaps, unstated assumptions) | Document + profile | Structural findings | Mid |

**Shared state: the claim ledger.** Agents do not pass text to each other. They read and write one structured ledger keyed by claim ID. Each entry holds the claim text and span, type, fact-check verdict, sources, rebuttals, and copy edits touching its span. This makes conflicts detectable by span overlap.

**Execution plan**

1. Orchestrator classifies the document and selects a review profile.
2. Claim extractor runs first; everything else depends on the ledger.
3. Fact-checker, devil's advocate, copy editor, originality and structure run in parallel.
4. Orchestrator runs conflict resolution over the ledger.
5. Orchestrator ranks findings and writes the report.

**Conflict resolution rules**

- Copy edit touches a span flagged wrong or unsupported → drop the copy edit; the fact finding wins.
- Devil's advocate attacks a claim the fact-checker verified → keep the rebuttal only if it targets the interpretation, not the fact; otherwise discard.
- Two agents flag the same span → merge into one finding with both reasons.
- Rebuttal with no retrieved source → downgrade to "consider" severity.
- Severity order: factual error > unsupported claim > strong rebuttal > structure > style.

**Budgets.** Each job has caps on tokens, search calls and wall-clock time. Agents degrade gracefully (for example, check the top 15 claims by importance) rather than fail.

## Review profiles and report format

v1 ships two profiles; the orchestrator picks one automatically unless the user names it.

| Profile | Agents | Emphasis |
| --- | --- | --- |
| Opinion / article | Extractor, fact-checker, devil's advocate, copy editor, originality | Facts right, strongest counter-case addressed |
| Technical design doc | Extractor, fact-checker, devil's advocate (argues against the design), structure reviewer, copy editor | Gaps, unstated assumptions, alternatives not considered |

Later candidates: report / analyst note (number consistency, source checks) and essay.

**Report structure**

1. **Verdict line:** one sentence on overall readiness, plus counts by severity.
2. **Must fix:** factual errors and unsupported claims, each with the quoted span, the problem, and sources.
3. **Strongest counter-case:** the devil's advocate's top 3 rebuttals, each with sources and the claim it targets.
4. **Should fix:** structure findings.
5. **Polish:** copy edits, grouped.
6. **Originality notes:** possible matches, labelled heuristic.
7. **Appendix:** full claim ledger with verdicts.

The report is generated as structured JSON, then rendered to HTML for email and to markdown for inline return.

## MCP interface

One tool, `review_document`, over Streamable HTTP with OAuth 2.1.

**Inputs**

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `content` | string | one of content/url | Document text or markdown; size-capped (e.g. 20k words) |
| `url` | string | one of content/url | Public page; fetched and extracted server-side |
| `profile` | enum | no | `auto` (default), `opinion`, `design_doc` |
| `focus` | string | no | Free-text steer, e.g. "check the financial figures" |

The API key and email never appear as tool arguments; they come from the signed-in account.

**Response modes**, chosen from the client's declared capabilities at call time:

1. **Tasks:** return a task handle; status updates carry the current step and percent; the result is the markdown report. See the [MCP Tasks spec](https://modelcontextprotocol.io/extensions/tasks/overview).
2. **Progress:** stream `notifications/progress` with step messages; return the markdown report at the end.
3. **Fire-and-forget:** return a short confirmation with a job ID and a note that the report is on its way by email.

In all modes the report is emailed, so a dropped connection loses nothing.

**Errors:** missing setup (returns a link to the setup page), invalid or out-of-credit provider key, document too long, URL unreachable, rate limit exceeded. Each returns a clear, human-readable message.

## Async jobs and email delivery

Every review is a queued job, even when the caller waits synchronously; the MCP layer only watches the job.

**Job lifecycle:** queued → extracting → reviewing → resolving → reporting → emailed → purged (or failed).

- **Queue:** a durable queue; workers pull jobs and run the orchestrator.
- **Progress:** workers write step, message and percent to the job record; the MCP layer relays them as task status or progress notifications.
- **Retries:** agent calls retry with backoff on transient errors (rate limits, timeouts). A failed agent degrades the report ("fact-check unavailable") rather than failing the job.
- **Timeout:** hard cap per job (e.g. 10 minutes); partial results are emailed with a note.
- **Idempotency:** a content hash plus user ID de-duplicates accidental double submits within a short window.
- **Email:** sent through a transactional email provider to the Google-verified address only. HTML report with a plain-text fallback. Sender domain set up with SPF and DKIM.
- **Failure email:** if the job fails, the user gets a short email with the reason and whether a retry is likely to help.

## Auth, API keys and security

Sign in with Google is the only identity provider; the MCP server acts as the OAuth authorization server toward clients and delegates login to Google.

**Auth flow**

1. Client connects; the server returns 401 with its OAuth metadata.
2. Client opens the authorize URL; the user signs in with Google.
3. Server creates or looks up the user by Google subject ID and verified email.
4. Server issues its own short-lived access token and refresh token to the client (never Google's tokens).

**API key handling**

- Entered only on the setup web page, never in chat or tool arguments.
- Validated with a minimal test call before saving.
- Encrypted at rest with envelope encryption: a per-user data key, wrapped by a master key in a managed KMS or secrets store.
- Decrypted only inside a worker for the duration of a job; never logged, never sent to the client.
- Users can rotate or delete their key, and delete their account, from the setup page.

**Other controls**

- **Prompt injection:** documents and fetched pages are untrusted data. Agents get no tools that act outside the job (no email, no writes). Only the report step emails, and only to the account's verified address.
- **URL fetching:** block private IP ranges and non-HTTP schemes (SSRF protection); cap size and redirects.
- **Rate limits:** per user (e.g. 20 reviews per day) and global, plus a per-job cost ceiling.
- **Transport:** HTTPS only; secure, HTTP-only session cookies on the setup page.

## Data model and retention

Only three tables persist; document content never outlives its job.

| Table | Fields | Retention |
| --- | --- | --- |
| `users` | id, google_sub, email, created_at, daily_count, last_review_at | Until account deletion |
| `provider_keys` | user_id, provider, encrypted_key, wrapped_data_key, created_at, last_validated_at | Until the user deletes it or the account |
| `jobs` | id, user_id, status, step, percent, profile, content_hash, created_at, finished_at, error | Metadata kept 30 days for debugging and cost stats; no content |

**Ephemeral (object storage or the queue payload)**

- Document text and fetched pages: encrypted, deleted when the job finishes.
- Claim ledger and agent outputs: held during the job, deleted after the report is emailed.
- Report: deleted after email delivery.

**OAuth tokens:** refresh tokens are stored hashed; access tokens are short-lived.

**Privacy promise (README):** "Your documents and reports are deleted after delivery. We store your email and your encrypted API key, nothing else."

## Evaluation

Quality is measured on a seeded-error set, and the headline results table goes in the README.

**Dataset**

- Start with 30 documents, growing to 100: blog posts (including your own drafts), op-eds, and design docs.
- For each, create a seeded copy with known defects and an answer key:
  - wrong numbers, dates or attributions (fact-checker);
  - unsupported claims (fact-checker);
  - a deliberately weak argument with a known strong rebuttal (devil's advocate);
  - borrowed sentences from a known source (originality);
  - grammar and clarity errors (copy editor).

**Metrics**

| Metric | What it measures |
| --- | --- |
| Recall per defect type | Share of seeded defects caught |
| Precision | Share of findings that are real (reviewed by hand, or by an LLM judge spot-checked by hand) |
| Citation validity | Share of cited sources that actually support the verdict |
| Rebuttal strength | LLM-judge score 1–5 against the known strongest rebuttal |
| Conflict-resolution accuracy | Share of seeded conflicts resolved per the rules |
| Cost and latency | Tokens, search calls and seconds per job |

**Baselines:** a single-model prompt ("review this and argue the other side") on the same set. The gap over that baseline is the core evidence that the multi-agent design earns its complexity.

**Regression:** the eval runs in CI on every change to prompts, agents or the orchestrator.

## Observability, cost and rate limits

Every job produces a trace, but traces record metadata and never document content.

- **Tracing:** one trace per job, with a span per agent call recording model, tokens in and out, latency, retries and the orchestrator's decisions (profile chosen, conflicts resolved and why). OpenTelemetry, exported to a hosted tracing backend's free tier.
- **Content redaction:** trace attributes hold claim IDs and counts, not text, to keep the privacy promise.
- **Cost:** inference is billed to the user's key. The service pays only for hosting, queue, storage, search and email. Target: under $10 a month at portfolio-level traffic, within free tiers where possible.
- **Per-job cost estimate:** shown in the report footer (tokens used, estimated cost on the user's key).
- **Rate limits:** per-user daily cap, per-job token and search caps, and a global circuit breaker on the search provider.
- **Dashboards:** jobs per day, p50 and p95 latency, failure rate by agent, and search calls per job.

## Tech stack

Python throughout, with managed services everywhere except the agent layer, so build time goes into the multi-agent core. Candidates are not yet checked against current pricing; confirm free-tier limits before committing.

| Layer | Choice | Why |
| --- | --- | --- |
| MCP server | Official MCP Python SDK, Streamable HTTP | First-party support for Tasks, progress and OAuth |
| Web and setup page | FastAPI plus a minimal HTML page | Same process as the MCP server; small surface |
| Orchestration | Plain Python asyncio with typed ledger models (Pydantic) | Hand-built orchestration shows the design; a framework hides it |
| Model access | Thin provider adapter (Anthropic first, OpenAI second) | Bring-your-own-key across providers |
| Queue and workers | Postgres-backed job queue, or Redis with a worker library | One fewer service if Postgres |
| Database | Managed Postgres | Users, keys, job metadata |
| Secrets and encryption | Cloud KMS or a secrets manager for the master key | Envelope encryption |
| Search and retrieval | A search API with a free tier, plus page fetch and extraction | Evidence for the fact-checker and devil's advocate |
| Email | Transactional email provider | Deliverability, SPF and DKIM |
| Hosting | Container platform with scale-to-zero | Near-zero idle cost |
| Tracing | OpenTelemetry to a hosted backend | Free tier; standard |
| CI | GitHub Actions: tests plus the eval subset | Public repo |

## Build phases

Five phases, each with a gate; no hosting work starts until evidence retrieval proves itself in Phase 0.

1. **Phase 0 · Evidence prototype.** Extractor + devil's advocate script on your own drafts; pick a search API; publish 2–3 posts by hand.
   - Gate: rebuttals cite real, relevant sources on your drafts.
2. **Phase 1 · Core engine (local).** All agents, claim ledger, orchestrator with conflict rules, CLI; seeded eval set of 30 documents.
   - Gate: beats the single-prompt baseline on the eval set.
3. **Phase 2 · Local MCP server.** `review_document` over stdio, progress notifications, key from an environment variable.
   - Gate: used end to end on your own blog posts.
4. **Phase 3 · Hosted service.** Google OAuth, setup page, encrypted keys, queue and workers, email delivery, MCP Tasks.
   - Gate: full review from Claude and ChatGPT, report arrives by email.
5. **Phase 4 · Harden and launch.** Tracing, rate limits, eval in CI, README with results table, write-up post.

Phases 0–2 cost nothing to run and already make the tool useful for your blog; Phases 3–4 turn it into the portfolio version.

## Risks and open questions

The biggest risk is retrieval quality; it is prototyped first, in Phase 0.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Weak counter-evidence retrieval | The main differentiator fails | Prototype first; query rewriting per claim; primary-source preference |
| Infrastructure consumes the timeline | Multi-agent core gets thin | Managed services; strict phase gates |
| Client support for Tasks and progress varies | Poor experience in some clients | Email delivery in every mode |
| Bring-your-own-key friction | Few outside users | Accept; judge success on the eval and your own usage |
| Prompt injection via documents | Data leakage or misuse | Agents have no external actions; email only to the verified address |
| Search API free-tier limits | Jobs throttle or fail | Per-job search budget; degrade gracefully |
| Existing critique MCP servers | Looks derivative | Lead the README with evidence-backed results against the baseline |

**Open questions**

- [ ] Which search API: compare quality and free-tier limits in Phase 0.
- [ ] Offer a short-lived "view report online" link, or email only?
- [ ] Which clients must work at launch: Claude Desktop, claude.ai, ChatGPT, Cursor?
- [ ] Allow keys for more than one provider per user in v1?
- [ ] Product name.
