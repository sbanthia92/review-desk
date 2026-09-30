"""Local MCP server over stdio (T12): one tool, ``review_document``.

Run with ``python -m reviewdesk.mcp_local``. Build a server around any
pipeline with ``create_server(pipeline=..., fetcher=..., max_words=...)``.
See ``README.md`` in this package for Claude Desktop and Claude Code setup.
"""

from reviewdesk.mcp_local.errors import InvalidInput, user_message
from reviewdesk.mcp_local.main import main
from reviewdesk.mcp_local.server import DEFAULT_MAX_WORDS, TOOL_NAME, create_server
from reviewdesk.mcp_local.wiring import (
    REGISTRY_BUILDER,
    REGISTRY_MODULE,
    EnvPipeline,
    RegistryPipeline,
    ReviewRunner,
    adapt_pipeline,
    default_runner,
)

__all__ = [
    "DEFAULT_MAX_WORDS",
    "REGISTRY_BUILDER",
    "REGISTRY_MODULE",
    "TOOL_NAME",
    "EnvPipeline",
    "InvalidInput",
    "RegistryPipeline",
    "ReviewRunner",
    "adapt_pipeline",
    "create_server",
    "default_runner",
    "main",
    "user_message",
]
