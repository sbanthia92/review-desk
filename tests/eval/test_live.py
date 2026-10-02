from pathlib import Path

import pytest

from eval import live
from reviewdesk.contracts import MissingSetup


def test_load_env_fills_only_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env = tmp_path / ".env"
    env.write_text("REVIEWDESK_TEST_A=file\nREVIEWDESK_TEST_B=file\n")
    monkeypatch.setenv("REVIEWDESK_TEST_A", "process")
    monkeypatch.delenv("REVIEWDESK_TEST_B", raising=False)
    live.load_env(env)
    import os

    assert os.environ["REVIEWDESK_TEST_A"] == "process"
    assert os.environ["REVIEWDESK_TEST_B"] == "file"
    monkeypatch.delenv("REVIEWDESK_TEST_B")


def test_live_pipeline_without_keys_raises_missing_setup(monkeypatch: pytest.MonkeyPatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "TAVILY_API_KEY",
        "EXA_API_KEY",
    ):
        monkeypatch.setenv(var, "")
    monkeypatch.setattr(live, "ENV_FILE", Path("/nonexistent/.env"))
    monkeypatch.setattr(live, "load_env", lambda path=None: None)
    with pytest.raises(MissingSetup):
        live.LivePipeline().registry()
