from pathlib import Path

from reviewdesk.mcp_local.main import load_env


def test_env_file_fills_unset_variables_only(tmp_path: Path):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-file\nTAVILY_API_KEY=t\nEMPTY=\n")
    env = {"ANTHROPIC_API_KEY": "from-process", "TAVILY_API_KEY": ""}
    used = load_env(env, cwd=tmp_path)
    assert used == tmp_path / ".env"
    assert env["ANTHROPIC_API_KEY"] == "from-process"  # the process wins
    assert env["TAVILY_API_KEY"] == "t"  # blank counts as unset
    assert "EMPTY" not in env


def test_named_env_file_and_missing_file(tmp_path: Path):
    other = tmp_path / "keys.env"
    other.write_text("EXA_API_KEY=e\n")
    env = {"REVIEWDESK_ENV_FILE": str(other)}
    assert load_env(env, cwd=tmp_path) == other
    assert env["EXA_API_KEY"] == "e"
    assert load_env({}, cwd=tmp_path) is None  # no .env here: ignored
