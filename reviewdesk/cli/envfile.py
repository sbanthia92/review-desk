"""A tiny ``.env`` parser (no dependency).

Supports ``KEY=VALUE`` lines, ``export KEY=VALUE``, blank lines, ``#``
comments (whole-line, or after whitespace in an unquoted value), and single
or double quotes around the value. Double-quoted values understand ``\\n``,
``\\t``, ``\\"`` and ``\\\\``. No variable interpolation. Malformed lines are
skipped; their content is never echoed (it may hold a key).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}


def _unquote_double(body: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body) and body[i + 1] in _ESCAPES:
            out.append(_ESCAPES[body[i + 1]])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        body = value[1:-1]
        return _unquote_double(body) if value[0] == '"' else body
    if value and value[0] in "'\"":
        # Opening quote with trailing comment: KEY="v" # note
        quote = value[0]
        end = value.find(quote, 1)
        if end > 0:
            body = value[1:end]
            return _unquote_double(body) if quote == '"' else body
    comment = re.search(r"\s#", value)
    if comment:
        value = value[: comment.start()]
    return value.strip()


def parse_env(text: str) -> dict[str, str]:
    """Parse ``.env`` text into a dict (later duplicates win)."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        name, sep, rest = stripped.partition("=")
        name = name.strip()
        if not sep or not _NAME.match(name):
            continue
        values[name] = _parse_value(rest)
    return values


def load_env_file(path: Path) -> dict[str, str]:
    """Read and parse ``path``. Raises ``OSError`` if it cannot be read."""
    return parse_env(path.read_text(encoding="utf-8-sig"))


def merge_env(file_values: Mapping[str, str], environ: Mapping[str, str]) -> dict[str, str]:
    """``file_values`` overlaid by ``environ``; set, non-blank process variables win."""
    merged = dict(file_values)
    for name, value in environ.items():
        if value.strip() or name not in merged:
            merged[name] = value
    return merged
