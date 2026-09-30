# Contract change requests

Tasks never edit `reviewdesk/contracts/`, `reviewdesk/testing/` or root config
directly. Append a request here instead and work around it with a local adapter
inside your owned paths. The lead resolves requests between batches.

Template (copy, fill in, append below the line):

```
## <task id> · <short title>
- **What:** the exact type, field, method or dependency to add or change.
- **Why:** what breaks or gets awkward without it.
- **Workaround in place:** the local adapter you used, and where.
- **Status:** open
```

---

## T3 · Non-fatal notes on AgentResult
- **What:** add `notes: list[str] = Field(default_factory=list)` to `AgentResult` (non-fatal degradation notices, e.g. "2 quoted claims could not be located and were dropped").
- **Why:** `error` means the agent degraded; there is no field for "succeeded, but dropped/merged some output". The orchestrator could copy these into `Report.notes`.
- **Workaround in place:** the extractor puts the counts in its final `ProgressEvent.message` and exposes them via `ExtractorAgent.extract_claims(ctx) -> ExtractionOutcome.notes()` in `reviewdesk/agents/extractor/agent.py`.
- **Status:** accepted — `AgentResult.notes` added on main (lead). T3 may switch to it in a follow-up.

## T2 · List search provider keys in `.env.example`
- **What:** add `BRAVE_SEARCH_API_KEY=`, `TAVILY_API_KEY=` and `EXA_API_KEY=` to `.env.example` (replacing the "added once T2 picks providers" comment).
- **Why:** `reviewdesk.providers.search.config.search_clients_from_env` and `scripts/compare_search.py` read these names; users need to know them.
- **Workaround in place:** names are documented in `reviewdesk/providers/search/__init__.py` and `config.py` docstrings; `compare_search.py --env-file` reads any `.env`.
- **Status:** accepted — keys added to `.env.example` on main (lead).

## T12 · `focus` on the `ReviewPipeline` protocol
- **What:** add `*, focus: str | None = None` to `ReviewPipeline.__call__` in `reviewdesk/contracts/interfaces.py` (`reviewdesk.pipeline.run_review` already accepts it).
- **Why:** the MCP tool's `focus` argument (DESIGN.md "MCP interface") has to reach the pipeline; the contract shape has no way to pass it, so any injected `ReviewPipeline` silently drops it.
- **Workaround in place:** `reviewdesk/mcp_local/wiring.py` defines a local `ReviewRunner` protocol (contract shape + `focus`) and `adapt_pipeline()` to wrap a plain `ReviewPipeline`.
- **Status:** declined (lead) — `reviewdesk.pipeline.run_review` already takes `focus` as a keyword; adding it to the protocol would make simpler pipelines (eval fakes, baselines) non-conforming. Keep the local adapter.

## T12 · Surface rejected provider keys from `run_review`
- **What:** have the orchestrator (T10) re-raise `AuthError` (at least from the LLM client) instead of turning it into per-agent degradation notes; or add a `Report` field listing provider errors by type.
- **Why:** with an invalid or out-of-credit key every agent degrades and the user gets an empty "ready" report instead of the "invalid key" error DESIGN.md requires.
- **Workaround in place:** `RegistryPipeline` in `reviewdesk/mcp_local/wiring.py` wraps the registry's `llm` and `search` in proxies that record the first `AuthError` and raise it after `run_review` returns. T11's CLI and T16's workers will need the same unless the orchestrator does it.
- **Status:** accepted (lead) — `Orchestrator.review` now wraps the LLM and search clients per job (`reviewdesk/orchestrator/authwatch.py`) and raises the first `AuthError` at the end of the job. T12's proxies are now redundant but harmless.

## T1 · Declare `httpx2` explicitly
- **What:** add `httpx2` (the HTTP library `anthropic>=1.10` and `openai>=3.22` are built on, currently 2.13.1) as a direct dependency or dev dependency in `pyproject.toml`.
- **Why:** the LLM adapter tests replay recorded fixtures through `httpx2.MockTransport` injected into the SDK clients (`respx` only patches `httpx`, which the SDKs no longer use). Adapter code imports `httpx2` only under `TYPE_CHECKING`. Today it resolves only as a transitive dependency.
- **Workaround in place:** tests import the transitive `httpx2` directly (`tests/providers/llm/replay.py`).
- **Status:** accepted — `httpx2` added as a dev dependency on main (lead).
