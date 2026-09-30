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
- **Status:** open
