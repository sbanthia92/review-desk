"""``reviewdesk review`` end to end on fake-backed registries (offline)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from reviewdesk.cli import run
from reviewdesk.cli.errors import (
    EXIT_AUTH,
    EXIT_FAILED,
    EXIT_FETCH,
    EXIT_INPUT,
    EXIT_OK,
    EXIT_RATE_LIMIT,
    EXIT_SETUP,
    EXIT_TOO_LONG,
    EXIT_USAGE,
)
from reviewdesk.contracts import (
    AddClaim,
    AgentName,
    AuthError,
    BlockedURLError,
    Finding,
    MissingSetup,
    Profile,
    RateLimitError,
    Report,
    ReviewContext,
    Severity,
)
from reviewdesk.orchestrator import AgentRegistry
from reviewdesk.registry import build_registry
from reviewdesk.testing.fakes import (
    OPINION_DOC,
    FakeAgent,
    FakeFetcher,
    FakeLLM,
    FakeSearch,
    sample_ledger,
)

SECRET = "sk-ant-SECRET-never-print"
PAGE_URL = "https://example.org/post"


class Harness:
    """Builds fake registries, records what the pipeline saw."""

    def __init__(
        self,
        *,
        llm: FakeLLM | None = None,
        pages: dict[str, Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.llm = llm or FakeLLM()
        self.fetcher = FakeFetcher(pages or {})
        self.raises = raises
        self.contexts: list[ReviewContext] = []
        self.envs: list[Mapping[str, str]] = []
        self.closed = 0

    def _copyedit(self, ctx: ReviewContext) -> Any:
        from reviewdesk.contracts import AgentResult

        self.contexts.append(ctx)
        return AgentResult(
            agent=AgentName.COPYEDIT,
            findings=[
                Finding(
                    id="find_cli_style",
                    agent=AgentName.COPYEDIT,
                    severity=Severity.STYLE,
                    message="Tighten this sentence.",
                    suggestion="shorter",
                )
            ],
        )

    def factory(self, env: Mapping[str, str]) -> AgentRegistry:
        self.envs.append(env)
        if self.raises is not None:
            raise self.raises
        harness = self

        class ClosableSearch(FakeSearch):
            async def aclose(self) -> None:
                harness.closed += 1

        agents = [
            FakeAgent(
                AgentName.EXTRACTOR,
                ledger_updates=[AddClaim(claim=e.claim) for e in sample_ledger().entries.values()],
                progress=["extracting claims (fake)"],
            ),
            FakeAgent(AgentName.COPYEDIT, on_run=self._copyedit),
        ]
        return build_registry(
            llm=self.llm, search=ClosableSearch(), fetcher=self.fetcher, agents=agents
        )


def cli(
    argv: list[str], harness: Harness, env: Mapping[str, str] | None = None
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = run(
        [*argv, "--no-env-file"] if "--env-file" not in argv else argv,
        registry_factory=harness.factory,
        environ=env if env is not None else {},
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def draft(tmp_path: Path) -> Path:
    path = tmp_path / "draft.md"
    path.write_text(OPINION_DOC.text, encoding="utf-8")
    return path


def _progress_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("[")]


def test_file_review_prints_markdown_and_progress(draft: Path) -> None:
    harness = Harness()
    code, out, err = cli(["review", str(draft), "--profile", "opinion"], harness)
    assert code == EXIT_OK, err
    assert out.startswith("# Review Desk report")
    assert "Tighten this sentence." in out
    lines = _progress_lines(err)
    assert lines, err
    percents = [float(line[1:4]) for line in lines]
    assert percents == sorted(percents)
    assert lines[-1].startswith("[100%] done")
    assert any("extracting claims (fake)" in line for line in lines)
    assert "# Review Desk report" not in err
    assert harness.contexts[0].profile == Profile.OPINION
    assert harness.contexts[0].document.text == OPINION_DOC.text
    assert harness.closed == 1


def test_progress_order_follows_the_lifecycle(draft: Path) -> None:
    code, _out, err = cli(["review", str(draft)], Harness())
    assert code == EXIT_OK
    steps = [line.split("] ", 1)[1].split(":", 1)[0] for line in _progress_lines(err)]
    order = ["classifying", "extracting", "planning", "reviewing", "resolving", "reporting", "done"]
    positions = [steps.index(step) for step in order]
    assert positions == sorted(positions)


def test_json_output(draft: Path) -> None:
    code, out, _err = cli(["review", str(draft), "--json", "--quiet"], Harness())
    assert code == EXIT_OK
    report = Report.model_validate_json(out)
    assert any(f.message == "Tighten this sentence." for f in report.polish)
    assert json.loads(out)["document_id"] == report.document_id


def test_quiet_suppresses_progress(draft: Path) -> None:
    code, out, err = cli(["review", str(draft), "--quiet"], Harness())
    assert code == EXIT_OK
    assert err == ""
    assert out.startswith("# Review Desk report")


def test_focus_and_style_guide_reach_the_agents(draft: Path, tmp_path: Path) -> None:
    guide = tmp_path / "style.md"
    guide.write_text("Use British spelling.\n", encoding="utf-8")
    harness = Harness()
    code, _out, err = cli(
        [
            "review",
            str(draft),
            "--focus",
            "  check the   numbers ",
            "--style-guide",
            str(guide),
            "--quiet",
        ],
        harness,
    )
    assert code == EXIT_OK, err
    ctx = harness.contexts[0]
    assert ctx.focus == "check the numbers"
    assert ctx.style_guide == "Use British spelling."


def test_url_input_uses_the_registry_fetcher() -> None:
    harness = Harness(pages={PAGE_URL: OPINION_DOC.text})
    code, out, err = cli(["review", PAGE_URL], harness)
    assert code == EXIT_OK, err
    assert harness.fetcher.fetched == [PAGE_URL]
    assert harness.contexts[0].document.source_url == PAGE_URL
    assert _progress_lines(err)[0].startswith("[  0%] fetching")
    assert out.startswith("# Review Desk report")


def test_blocked_url() -> None:
    url = "http://127.0.0.1/admin"
    harness = Harness(pages={url: BlockedURLError("private address")})
    code, out, err = cli(["review", url], harness)
    assert code == EXIT_FETCH
    assert out == ""
    assert "refused to fetch" in err
    assert "Traceback" not in err
    assert harness.closed == 1


def test_unreachable_url() -> None:
    code, out, err = cli(["review", "https://nowhere.example/x"], Harness())
    assert code == EXIT_FETCH
    assert "URL unreachable" in err
    assert out == ""


def test_empty_page() -> None:
    code, _out, err = cli(["review", PAGE_URL], Harness(pages={PAGE_URL: "   "}))
    assert code == EXIT_INPUT
    assert "no readable text" in err


def test_non_http_scheme_is_rejected() -> None:
    code, _out, err = cli(["review", "file:///etc/passwd"], Harness())
    assert code == EXIT_INPUT
    assert "http(s)" in err


def test_file_not_found_does_not_need_keys(tmp_path: Path) -> None:
    harness = Harness(raises=MissingSetup("no keys"))
    code, out, err = cli(["review", str(tmp_path / "missing.md")], harness)
    assert code == EXIT_INPUT
    assert "file not found" in err
    assert out == ""
    assert harness.envs == []


def test_directory_and_empty_and_binary_files(tmp_path: Path) -> None:
    code, _o, err = cli(["review", str(tmp_path)], Harness())
    assert code == EXIT_INPUT and "not a file" in err
    empty = tmp_path / "empty.md"
    empty.write_text("  \n", encoding="utf-8")
    code, _o, err = cli(["review", str(empty)], Harness())
    assert code == EXIT_INPUT and "empty" in err
    binary = tmp_path / "blob.md"
    binary.write_bytes(b"\xff\xfe\x00\x81bad")
    code, _o, err = cli(["review", str(binary)], Harness())
    assert code == EXIT_INPUT and "UTF-8" in err


def test_missing_style_guide(draft: Path, tmp_path: Path) -> None:
    code, _o, err = cli(["review", str(draft), "--style-guide", str(tmp_path / "x")], Harness())
    assert code == EXIT_INPUT
    assert "style guide not found" in err


def test_document_too_long(tmp_path: Path) -> None:
    path = tmp_path / "long.md"
    secret_text = "confidentialword " * 30
    path.write_text(secret_text, encoding="utf-8")
    harness = Harness()
    code, out, err = cli(["review", str(path), "--max-words", "10"], harness)
    assert code == EXIT_TOO_LONG
    assert "30 words; the limit is 10 words" in err
    assert "confidentialword" not in err
    assert out == ""
    assert harness.envs == []  # rejected before any provider is built


def test_word_cap_from_env_and_default(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("word " * 30, encoding="utf-8")
    code, _o, err = cli(["review", str(path)], Harness(), env={"REVIEWDESK_MAX_WORDS": "5"})
    assert code == EXIT_TOO_LONG, err
    code, _o, _e = cli(["review", str(path), "--quiet"], Harness())
    assert code == EXIT_OK


def test_word_cap_applies_to_urls() -> None:
    harness = Harness(pages={PAGE_URL: "word " * 50})
    code, _o, err = cli(["review", PAGE_URL, "--max-words", "20"], harness)
    assert code == EXIT_TOO_LONG
    assert "50 words" in err
    assert harness.closed == 1


def test_bad_max_words(draft: Path) -> None:
    code, _o, err = cli(["review", str(draft), "--max-words", "0"], Harness())
    assert code == EXIT_INPUT
    code, _o, err = cli(["review", str(draft), "--max-words", "lots"], Harness())
    assert code == EXIT_USAGE
    assert "invalid int" in err


def test_missing_setup(draft: Path) -> None:
    harness = Harness(raises=MissingSetup("no LLM key: set one of ANTHROPIC_API_KEY"))
    code, out, err = cli(["review", str(draft)], harness)
    assert code == EXIT_SETUP
    assert "Missing setup" in err
    assert "ANTHROPIC_API_KEY" in err
    assert out == ""


def test_real_factory_missing_setup(draft: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    code = run(["review", str(draft), "--no-env-file"], environ={}, stdout=out, stderr=err)
    assert code == EXIT_SETUP
    assert "BRAVE_SEARCH_API_KEY" in err.getvalue()
    assert "Traceback" not in err.getvalue()


def test_auth_error_from_the_provider(draft: Path) -> None:
    harness = Harness(llm=FakeLLM(default=AuthError("anthropic rejected the API key")))
    code, out, err = cli(["review", str(draft)], harness)
    assert code == EXIT_AUTH
    assert "Invalid or out-of-credit provider key" in err
    assert out == ""
    assert harness.closed == 1


def test_rate_limit(draft: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import reviewdesk.pipeline

    async def limited(*args: Any, **kwargs: Any) -> Report:
        raise RateLimitError("brave: 429")

    monkeypatch.setattr(reviewdesk.pipeline, "run_review", limited)
    code, _out, err = cli(["review", str(draft)], Harness())
    assert code == EXIT_RATE_LIMIT
    assert "Rate limit exceeded" in err


def test_unexpected_error_has_no_traceback(draft: Path) -> None:
    code, _out, err = cli(["review", str(draft)], Harness(raises=RuntimeError(SECRET)))
    assert code == EXIT_FAILED
    assert "RuntimeError" in err
    assert SECRET not in err
    assert "Traceback" not in err


def test_env_file_is_loaded_and_process_env_wins(draft: Path, tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        f"# keys\nexport ANTHROPIC_API_KEY='{SECRET}'\nBRAVE_SEARCH_API_KEY=from-file\n"
        "REVIEWDESK_MAX_WORDS=100000 # big\n",
        encoding="utf-8",
    )
    harness = Harness()
    out, err = io.StringIO(), io.StringIO()
    code = run(
        ["review", str(draft), "--env-file", str(env_file), "--quiet"],
        registry_factory=harness.factory,
        environ={"BRAVE_SEARCH_API_KEY": "from-process", "OPENAI_API_KEY": ""},
        stdout=out,
        stderr=err,
    )
    assert code == EXIT_OK, err.getvalue()
    env = harness.envs[0]
    assert env["ANTHROPIC_API_KEY"] == SECRET
    assert env["BRAVE_SEARCH_API_KEY"] == "from-process"
    assert env["REVIEWDESK_MAX_WORDS"] == "100000"
    assert SECRET not in out.getvalue() + err.getvalue()


def test_default_env_file_in_cwd(
    draft: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("TAVILY_API_KEY=abc\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    harness = Harness()
    out, err = io.StringIO(), io.StringIO()
    code = run(
        ["review", str(draft), "--quiet"],
        registry_factory=harness.factory,
        environ={},
        stdout=out,
        stderr=err,
    )
    assert code == EXIT_OK
    assert harness.envs[0]["TAVILY_API_KEY"] == "abc"
    code = run(
        ["review", str(draft), "--quiet", "--no-env-file"],
        registry_factory=harness.factory,
        environ={},
        stdout=out,
        stderr=err,
    )
    assert "TAVILY_API_KEY" not in harness.envs[1]


def test_missing_env_file(draft: Path, tmp_path: Path) -> None:
    code, _o, err = cli(["review", str(draft), "--env-file", str(tmp_path / "nope.env")], Harness())
    assert code == EXIT_INPUT
    assert "env file not found" in err


def test_usage_errors() -> None:
    harness = Harness()
    out, err = io.StringIO(), io.StringIO()
    assert run([], registry_factory=harness.factory, stdout=out, stderr=err) == EXIT_USAGE
    assert "review" in err.getvalue()
    code, _o, err_text = cli(["review", "x.md", "--profile", "poem"], harness)
    assert code == EXIT_USAGE
    assert "invalid choice" in err_text
    out, err = io.StringIO(), io.StringIO()
    assert run(["review", "--help"], stdout=out, stderr=err) == EXIT_OK


def test_fake_flag_runs_offline_demo(draft: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    code = run(
        ["review", str(draft), "--fake", "--no-env-file"], environ={}, stdout=out, stderr=err
    )
    assert code == EXIT_OK, err.getvalue()
    assert "Demo mode" in out.getvalue()


def test_python_dash_m_entry_point(draft: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reviewdesk.cli",
            "review",
            str(draft),
            "--fake",
            "--no-env-file",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("# Review Desk report")
    missing = subprocess.run(
        [sys.executable, "-m", "reviewdesk.cli", "review", "/no/such/file.md", "--no-env-file"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert missing.returncode == EXIT_INPUT
    assert "Traceback" not in missing.stderr
