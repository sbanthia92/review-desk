"""The tiny ``.env`` parser."""

from __future__ import annotations

from reviewdesk.cli.envfile import merge_env, parse_env


def test_parse_env_forms() -> None:
    text = "\n".join(
        [
            "# comment",
            "",
            "PLAIN=value",
            "SPACED = spaced value  ",
            "export EXPORTED=yes",
            "SINGLE='a # not a comment'",
            'DOUBLE="line\\nnext \\"q\\""',
            'TRAILING="quoted" # note',
            "HASH=abc # comment",
            "EMPTY=",
            "not a line",
            "1BAD=x",
            "PLAIN=second",
        ]
    )
    assert parse_env(text) == {
        "PLAIN": "second",
        "SPACED": "spaced value",
        "EXPORTED": "yes",
        "SINGLE": "a # not a comment",
        "DOUBLE": 'line\nnext "q"',
        "TRAILING": "quoted",
        "HASH": "abc",
        "EMPTY": "",
    }


def test_merge_env_prefers_non_blank_process_values() -> None:
    merged = merge_env({"A": "file", "B": "file", "C": "file"}, {"A": "proc", "B": " ", "D": ""})
    assert merged == {"A": "proc", "B": "file", "C": "file", "D": ""}
