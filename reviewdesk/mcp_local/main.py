"""stdio entry point for the local MCP server.

Logs go to stderr only: stdout carries the MCP protocol. Configuration comes
from environment variables. They can be set in the MCP client's ``env`` block,
or kept in a ``.env`` file: the server loads ``REVIEWDESK_ENV_FILE`` if set,
otherwise ``.env`` in its working directory (the repo, when started with
``uv run --directory``). Variables already set in the process win, so keys
need not be copied into the client's configuration.

- provider and search keys, read by ``reviewdesk.registry.build_registry_from_env``;
- ``REVIEWDESK_FAKE=1`` to run the offline demo pipeline (no keys needed);
- ``REVIEWDESK_MAX_WORDS`` to change the 20,000-word cap;
- ``REVIEWDESK_LOG_LEVEL`` (default ``INFO``).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from reviewdesk.cli.envfile import load_env_file
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


ENV_FILE_VAR = "REVIEWDESK_ENV_FILE"


def load_env(environ: dict[str, str] | None = None, cwd: Path | None = None) -> Path | None:
    """Fill unset (or blank) variables from a ``.env`` file. Returns the file used.

    Uses ``REVIEWDESK_ENV_FILE`` if set, else ``.env`` in ``cwd``. Values are
    never logged. A missing or unreadable file is ignored.
    """
    env = os.environ if environ is None else environ
    named = env.get(ENV_FILE_VAR, "").strip()
    path = Path(named).expanduser() if named else (cwd or Path.cwd()) / ".env"
    if not path.is_file():
        return None
    try:
        values = load_env_file(path)
    except OSError:
        return None
    for key, value in values.items():
        if value and not env.get(key):
            env[key] = value
    return path


def main() -> None:
    """Run ``review_document`` over stdio until the client disconnects."""
    env_file = load_env()
    configure_logging()
    log = logging.getLogger("reviewdesk.mcp_local")
    if env_file is not None:
        log.info("loaded settings from %s", env_file.name)
    max_words = max_words_from_env(DEFAULT_MAX_WORDS)
    server = create_server(max_words=max_words)
    log.info(
        "Review Desk MCP server starting on stdio (%s mode, max %d words)",
        "demo" if fake_mode() else "real",
        max_words,
    )
    server.run("stdio")
