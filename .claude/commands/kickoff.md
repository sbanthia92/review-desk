---
description: Start the Review Desk build as lead — run Wave 0, then Wave 1 in parallel sub-agents
---

You are the lead for the Review Desk build. Read TASKS.md, DESIGN.md and CLAUDE.md first.

1. Run Wave 0 (T0) yourself on a branch `task/t0-foundation`. Make sure `uv run ruff check`, `uv run mypy` and `uv run pytest` pass, then merge into `main`.
2. Launch Wave 1 (T1–T10) with the `task-implementer` sub-agent: one task per sub-agent, each with worktree isolation on branch `task/<id>-<slug>`, at most 5 running at a time. Give each sub-agent its task section from TASKS.md verbatim, its owned paths, and the rule that it must not edit files outside them.
3. As each sub-agent finishes, review its diff against the task's "Done when", run the full test suite on the merged result, and merge. Resolve any CONTRACT_CHANGES.md requests before launching the next batch.
4. Run Wave 2 (T11, T12) the same way.
5. Stop. Report what was built, test status, open contract changes, and anything that needs my decision, so I can run the Phase 0 gate.

Commit messages: `T<id>: <summary>`. Do not push to a remote unless I ask.
