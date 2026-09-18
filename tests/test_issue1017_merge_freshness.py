"""#1017 — a PR's green marks describe the main it was merged into when the run was created.

GitHub never re-evaluates them. ``strict: false`` means those marks still satisfy
the merge gate. The check that closes that hole is option 2 on the issue: fail
when the live merge-base is more than N commits behind main, plus ruff F821 on
a locally rebuilt merge (the #763 NameError is a 1-commit deletion that any
N > 0 window would still allow). The 13-job matrix is not re-run; a false-red
test cell still needs a push, which CONTRIBUTING now says.

No test here talks to GitHub. HTTP is a fake opener. Git fixtures are local.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from scripts.merge_freshness import (
    CHECK_NAME,
    MAX_BEHIND,
    Conclusion,
    commits_behind,
    decide,
    evaluate_refs,
    list_open_pr_heads,
    main,
    post_check_run,
    run_f821,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "merge-freshness.yml"
SCRIPT = ROOT / "scripts" / "merge_freshness.py"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
CI = ROOT / ".github" / "workflows" / "ci.yml"

_ADAPTER = """def _for_terminal(text):
    return text


def other():
    return 1
"""

_ADAPTER_DELETED = """def other():
    return 1
"""

_ADAPTER_NEW_CALLER = """def _for_terminal(text):
    return text


def other():
    return 1


def audit():
    return _for_terminal("ok")
"""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "soup-test"
    env["GIT_AUTHOR_EMAIL"] = "soup@test"
    env["GIT_COMMITTER_NAME"] = "soup-test"
    env["GIT_COMMITTER_EMAIL"] = "soup@test"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )


def _init_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "soup@test")
    _git(repo, "config", "user.name", "soup-test")
    adapters = repo / "src" / "soup_cli" / "commands"
    adapters.mkdir(parents=True)
    (adapters / "adapters.py").write_text(_ADAPTER, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base with _for_terminal")
    return repo


def _commit_file(repo: Path, rel: str, contents: str, message: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    _git(repo, "add", rel)
    _git(repo, "commit", "-m", message)


class TestDecide:
    def test_within_the_cap_is_success(self):
        verdict = decide(behind=MAX_BEHIND)
        assert verdict.conclusion is Conclusion.SUCCESS
        assert verdict.exit_code == 0

    def test_one_past_the_cap_is_failure(self):
        """The cap is exclusive: N is allowed, N+1 is not."""
        verdict = decide(behind=MAX_BEHIND + 1)
        assert verdict.conclusion is Conclusion.FAILURE
        assert verdict.exit_code == 1
        assert str(MAX_BEHIND + 1) in verdict.title
        assert "gh run rerun" in verdict.summary

    def test_f821_fails_even_when_inside_the_cap(self):
        """#763 is a 1-commit deletion. A lag window of 10 would still merge it."""
        hit = (
            "src/soup_cli/commands/adapters.py:3:12: "
            "F821 Undefined name `_for_terminal`"
        )
        verdict = decide(behind=1, f821_hits=(hit,))
        assert verdict.conclusion is Conclusion.FAILURE
        assert "undefined names" in verdict.title
        assert "_for_terminal" in verdict.summary

    def test_conflict_fails_even_when_inside_the_cap(self):
        verdict = decide(behind=0, merge_conflict=True)
        assert verdict.conclusion is Conclusion.FAILURE
        assert "conflicts" in verdict.title

    def test_fetch_failed_is_neutral_not_failure(self):
        verdict = decide(behind=0, fetch_failed=True)
        assert verdict.conclusion is Conclusion.NEUTRAL
        assert verdict.exit_code == 0

    def test_negative_behind_is_rejected(self):
        with pytest.raises(ValueError, match="behind"):
            decide(behind=-1)

    def test_cli_decide_matches_the_function(self):
        assert main(["decide", "--behind", "0"]) == 0
        assert main(["decide", "--behind", str(MAX_BEHIND + 1)]) == 1
        assert main(["decide", "--behind", "0", "--fetch-failed"]) == 0


class TestThe763Shape:
    """Reconstruct the near-miss: branch before 0ec39268, add a caller, main deletes the name."""

    def test_live_merge_reports_the_undefined_name(self, tmp_path):
        repo = _init_repo(tmp_path)
        _git(repo, "checkout", "-b", "pr")
        _commit_file(
            repo,
            "src/soup_cli/commands/adapters.py",
            _ADAPTER_NEW_CALLER,
            "add a _for_terminal caller",
        )
        _git(repo, "checkout", "main")
        _commit_file(
            repo,
            "src/soup_cli/commands/adapters.py",
            _ADAPTER_DELETED,
            "delete _for_terminal",
        )
        assert commits_behind(repo, "main", "pr") == 1
        verdict = evaluate_refs(repo, "main", "pr")
        assert verdict.conclusion is Conclusion.FAILURE
        assert "_for_terminal" in verdict.summary or "undefined" in verdict.title.lower()

    def test_eleven_dummy_commits_trip_the_lag_cap(self, tmp_path):
        repo = _init_repo(tmp_path)
        _git(repo, "branch", "pr")
        for i in range(MAX_BEHIND + 1):
            _commit_file(repo, f"docs/pad-{i}.md", f"{i}\n", f"pad {i}")
        verdict = evaluate_refs(repo, "main", "pr")
        assert verdict.conclusion is Conclusion.FAILURE
        assert str(MAX_BEHIND + 1) in verdict.summary

    def test_ten_dummy_commits_stay_inside_the_cap(self, tmp_path):
        repo = _init_repo(tmp_path)
        _git(repo, "branch", "pr")
        for i in range(MAX_BEHIND):
            _commit_file(repo, f"docs/pad-{i}.md", f"{i}\n", f"pad {i}")
        verdict = evaluate_refs(repo, "main", "pr", f821_runner=_clean_ruff)
        assert verdict.conclusion is Conclusion.SUCCESS
        assert commits_behind(repo, "main", "pr") == MAX_BEHIND


class TestRunF821:
    def test_injected_ruff_output_is_parsed(self, tmp_path):
        (tmp_path / "src" / "soup_cli").mkdir(parents=True)

        def runner(argv, *, cwd):
            assert "--select" in argv and "F821" in argv
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout=(
                    "src/soup_cli/commands/adapters.py:1817:12: "
                    "F821 Undefined name `_for_terminal`\n"
                ),
                stderr="",
            )

        hits = run_f821(tmp_path, runner=runner)
        assert len(hits) == 1
        assert "_for_terminal" in hits[0]

    def test_no_targets_means_clean(self, tmp_path):
        assert run_f821(tmp_path) == ()


class TestChecksApi:
    def test_post_check_run_posts_the_named_check(self):
        captured: list[Request] = []

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def opener(request, timeout=30):
            captured.append(request)
            return _Resp(b"{}")

        verdict = decide(behind=0)
        post_check_run(
            repo="MakazhanAlpamys/Soup",
            sha="abc123",
            verdict=verdict,
            token="t",
            opener=opener,
        )
        assert len(captured) == 1
        request = captured[0]
        assert request.full_url.endswith("/repos/MakazhanAlpamys/Soup/check-runs")
        body = json.loads(request.data.decode("utf-8"))
        assert body["name"] == CHECK_NAME
        assert body["head_sha"] == "abc123"
        assert body["conclusion"] == "success"

    def test_list_open_pr_heads_paginates(self):
        pages = [
            [{"number": 1, "head": {"sha": "aa"}}],
            [{"number": 2, "head": {"sha": "bb"}}],
            [],
        ]

        class _Resp:
            def __init__(self, payload):
                self._payload = json.dumps(payload).encode("utf-8")

            def read(self):
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def opener(request, timeout=30):
            page = int(str(request.full_url).rsplit("page=", 1)[-1])
            return _Resp(pages[page - 1])

        found = list_open_pr_heads(
            repo="MakazhanAlpamys/Soup",
            token="t",
            opener=opener,
            per_page=1,
        )
        assert found == [(1, "aa"), (2, "bb")]

    def test_http_error_is_an_http_error(self):
        def opener(request, timeout=30):
            raise HTTPError(request.full_url, 403, "Forbidden", {}, None)

        with pytest.raises(HTTPError):
            post_check_run(
                repo="MakazhanAlpamys/Soup",
                sha="abc",
                verdict=decide(behind=0),
                token="t",
                opener=opener,
            )


class TestWorkflowPins:
    def test_the_workflow_file_exists(self):
        assert WORKFLOW.is_file()

    def test_it_runs_on_pull_request_and_on_push_to_main(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "pull_request:" in text
        assert "branches: [main]" in text
        assert "workflow_dispatch:" in text

    def test_it_does_not_use_pull_request_target(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "pull_request_target" not in text

    def test_the_reeval_job_checks_out_the_triggering_ref_not_a_pr_head(self):
        """Running PR code on a main push would let a fork post checks as us."""
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "python scripts/merge_freshness.py" in text
        assert "reeval" in text
        assert "pull_request_target" not in text

    def test_permissions_can_post_checks_and_cannot_write_contents(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "checks: write" in text
        assert "contents: read" in text
        assert "contents: write" not in text

    def test_the_cap_in_the_workflow_matches_the_script(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        assert f"--max-behind {MAX_BEHIND}" in text

    def test_concurrency_is_untouched_in_ci_yml(self):
        """#1017's concurrency half is already on main; this PR must not edit it."""
        text = CI.read_text(encoding="utf-8")
        assert "github.event_name == 'pull_request' && github.ref || github.sha" in text


class TestContributingDocumentsTheTrap:
    def test_the_frozen_merge_sha_is_named(self):
        text = CONTRIBUTING.read_text(encoding="utf-8")
        assert "#1017" in text
        assert "gh run rerun" in text
        assert "frozen" in text.lower() or "pins" in text.lower()
        assert CHECK_NAME in text
        assert "strict" in text.lower()

    def test_the_script_module_is_the_one_definition_of_the_cap(self):
        src = SCRIPT.read_text(encoding="utf-8")
        assert src.count(f"MAX_BEHIND = {MAX_BEHIND}") == 1


def _clean_ruff(argv, *, cwd):
    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
