"""Shared helpers for the local MCP server tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcp import Client
from mcp.types import CallToolResult, TextContent

from reviewdesk.contracts import Document, Profile, ProgressCallback, ProgressEvent, Report
from reviewdesk.mcp_local import TOOL_NAME
from reviewdesk.testing.fakes import sample_report


@dataclass
class Call:
    doc: Document
    profile: Profile
    focus: str | None


@dataclass
class RecordingPipeline:
    """A ``ReviewRunner`` that records calls, emits progress and returns a report."""

    report: Report = field(default_factory=sample_report)
    progress: list[tuple[str, float]] = field(default_factory=list)
    raises: BaseException | None = None
    match_document: bool = True
    """Stamp the returned report with the reviewed document's id (as run_review does)."""
    calls: list[Call] = field(default_factory=list)
    returned: list[Report] = field(default_factory=list)

    async def __call__(
        self,
        doc: Document,
        profile: Profile,
        on_progress: ProgressCallback,
        *,
        focus: str | None = None,
    ) -> Report:
        self.calls.append(Call(doc, profile, focus))
        for message, percent in self.progress:
            on_progress(ProgressEvent(step="reviewing", message=message, percent=percent))
        if self.raises is not None:
            raise self.raises
        report = self.report
        if self.match_document:
            report = report.model_copy(update={"document_id": doc.id})
        self.returned.append(report)
        return report


@dataclass
class ProgressLog:
    events: list[tuple[float, float | None, str | None]] = field(default_factory=list)

    async def __call__(self, progress: float, total: float | None, message: str | None) -> None:
        self.events.append((progress, total, message))

    @property
    def messages(self) -> list[str | None]:
        return [m for _, _, m in self.events]


def text_of(result: CallToolResult) -> str:
    assert result.content, "empty tool result"
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


async def call(
    server: Any,
    arguments: dict[str, Any],
    *,
    progress: ProgressLog | None = None,
    mode: str = "legacy",
) -> CallToolResult:
    """Call ``review_document`` over an in-process client (JSON-RPC in legacy mode)."""
    async with Client(server, mode=mode) as client:
        return await client.call_tool(TOOL_NAME, arguments, progress_callback=progress)
