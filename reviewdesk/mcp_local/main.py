"""stdio entry point for the local MCP server.

Logs go to stderr only: stdout carries the MCP protocol. Configuration comes
from environment variables (set them in the MCP client's ``env`` block):

- provider and search keys, read by ``reviewdesk.registry.build_registry_from_env``;
- ``REVIEWDESK_FAKE=1`` to run the offline demo pipeline (no keys needed);
- ``REVIEWDESK_MAX_WORDS`` to change the 20,000-word cap;
- ``REVIEWDESK_LOG_LEVEL`` (default ``INFO``).
"""

from __future__ import annotations

import logging
import os
import sys

from reviewdesk.mcp_local.server import DEFAULT_MAX_WORDS, create_server
from reviewdesk.mcp_local.wiring import fake_mode, max_words_from_env


def configure_logging() -> None:
    """Send every log record to stderr (never stdout, the MCP channel)."""
    level = os.environ.get("REVIEWDESK_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level if level in logging.getLevelNamesMapping() else "INFO",
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    # Provider SDKs and httpx can log request URLs and headers at DEBUG.
    for noisy in ("httpx", "httpcore", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    """Run ``review_document`` over stdio until the client disconnects."""
    configure_logging()
    log = logging.getLogger("reviewdesk.mcp_local")
    max_words = max_words_from_env(DEFAULT_MAX_WORDS)
    server = create_server(max_words=max_words)
    log.info(
        "Review Desk MCP server starting on stdio (%s mode, max %d words)",
        "demo" if fake_mode() else "real",
        max_words,
    )
    server.run("stdio")
