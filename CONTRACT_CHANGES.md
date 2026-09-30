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

## T1 · Declare `httpx2` explicitly
- **What:** add `httpx2` (the HTTP library `anthropic>=1.10` and `openai>=3.22` are built on, currently 2.13.1) as a direct dependency or dev dependency in `pyproject.toml`.
- **Why:** the LLM adapter tests replay recorded fixtures through `httpx2.MockTransport` injected into the SDK clients (`respx` only patches `httpx`, which the SDKs no longer use). Adapter code imports `httpx2` only under `TYPE_CHECKING`. Today it resolves only as a transitive dependency.
- **Workaround in place:** tests import the transitive `httpx2` directly (`tests/providers/llm/replay.py`).
- **Status:** open
