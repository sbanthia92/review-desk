"""Real end-to-end review (network, real keys). Run with ``uv run pytest -m live``."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from reviewdesk.cli import run
from reviewdesk.cli.envfile import load_env_file, merge_env
from reviewdesk.contracts import MissingSetup, Report
from reviewdesk.registry import resolve_env_config
from reviewdesk.testing.fakes import SHORT_DOC

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env() -> dict[str, str]:
    env_file = REPO_ROOT / ".env"
    values = load_env_file(env_file) if env_file.is_file() else {}
    return merge_env(values, os.environ)


@pytest.mark.live
def test_real_review_of_a_short_document(tmp_path: Path) -> None:
    env = _env()
    try:
        resolve_env_config(env)
    except MissingSetup:
        pytest.skip("no LLM or search key configured")
    draft = tmp_path / "short.md"
    draft.write_text(SHORT_DOC.text, encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    code = run(
        ["review", str(draft), "--profile", "opinion", "--json", "--no-env-file"],
        environ=env,
        stdout=out,
        stderr=err,
    )
    assert code == 0, err.getvalue()
    report = Report.model_validate_json(out.getvalue())
    assert report.verdict_line
    assert "[100%] done" in err.getvalue()
