"""End to end over real stdio: ``python -m reviewdesk.mcp_local`` in demo mode.

Spawns the server as a subprocess (no network, no keys) and talks to it with
the SDK's stdio client, exactly as Claude Desktop and Claude Code do.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

from reviewdesk.mcp_local import TOOL_NAME
from reviewdesk.testing.fakes import OPINION_DOC
from tests.mcp_local.helpers import ProgressLog, text_of

REPO = Path(__file__).resolve().parents[2]


def params(**extra_env: str) -> StdioServerParameters:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "LANG", "SYSTEMROOT", "TMPDIR"}
    }
    env.update({"REVIEWDESK_FAKE": "1", "PYTHONPATH": str(REPO), **extra_env})
    return StdioServerParameters(
        command=sys.executable, args=["-m", "reviewdesk.mcp_local"], env=env, cwd=str(REPO)
    )


async def test_review_over_stdio_in_demo_mode() -> None:
    progress = ProgressLog()
    async with Client(params(), mode="legacy", read_timeout_seconds=60) as client:
        tools = (await client.list_tools()).tools
        assert [t.name for t in tools] == [TOOL_NAME]
        result = await client.call_tool(
            TOOL_NAME, {"content": OPINION_DOC.text}, progress_callback=progress
        )
        too_long = await client.call_tool(TOOL_NAME, {"content": "a b c d e f"})

    assert not result.is_error
    assert text_of(result).startswith("# Review Desk report")
    assert "Demo mode" in text_of(result)
    assert progress.events, "expected notifications/progress over stdio"
    assert progress.events[-1][0] == 100.0
    assert all(total == 100.0 for _, total, _ in progress.events)
    assert not too_long.is_error


async def test_word_cap_from_env_over_stdio() -> None:
    async with Client(
        params(REVIEWDESK_MAX_WORDS="5"), mode="legacy", read_timeout_seconds=60
    ) as client:
        result = await client.call_tool(TOOL_NAME, {"content": "a b c d e f"})
    assert result.is_error
    assert "Document too long: 6 words; the limit is 5 words" in text_of(result)
