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
