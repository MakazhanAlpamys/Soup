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
import yaml

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

    def test_one_past_the_cap_is_neutral_not_failure(self):
        """Lag is informational. A 10-commit cap at 27/day is strict with extra steps."""
        verdict = decide(behind=MAX_BEHIND + 1)
        assert verdict.conclusion is Conclusion.NEUTRAL
        assert verdict.exit_code == 0
        assert "informational" in verdict.title

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

    def test_fetch_failed_is_failure_because_neutral_passes_a_required_check(self):
        verdict = decide(behind=0, fetch_failed=True)
        assert verdict.conclusion is Conclusion.FAILURE
        assert verdict.exit_code == 1

    def test_git_error_is_failure_not_a_conflict(self):
        verdict = decide(behind=0, git_error=True)
        assert verdict.conclusion is Conclusion.FAILURE
        assert "conflict" not in verdict.title

    def test_negative_behind_is_rejected(self):
        with pytest.raises(ValueError, match="behind"):
            decide(behind=-1)

    def test_cli_decide_matches_the_function(self):
        assert main(["decide", "--behind", "0"]) == 0
        assert main(["decide", "--behind", str(MAX_BEHIND + 1)]) == 0
        assert main(["decide", "--behind", "0", "--fetch-failed"]) == 1
        assert main(["decide", "--behind", "0", "--git-error"]) == 1


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

    def test_eleven_dummy_commits_are_neutral_lag_not_failure(self, tmp_path):
        repo = _init_repo(tmp_path)
        _git(repo, "branch", "pr")
        for i in range(MAX_BEHIND + 1):
            _commit_file(repo, f"docs/pad-{i}.md", f"{i}\n", f"pad {i}")
        verdict = evaluate_refs(repo, "main", "pr", f821_runner=_clean_ruff)
        assert verdict.conclusion is Conclusion.NEUTRAL
        assert str(MAX_BEHIND + 1) in verdict.summary

    def test_ten_dummy_commits_stay_inside_the_cap(self, tmp_path):
        repo = _init_repo(tmp_path)
        _git(repo, "branch", "pr")
        for i in range(MAX_BEHIND):
            _commit_file(repo, f"docs/pad-{i}.md", f"{i}\n", f"pad {i}")
        verdict = evaluate_refs(repo, "main", "pr", f821_runner=_clean_ruff)
        assert verdict.conclusion is Conclusion.SUCCESS
        assert commits_behind(repo, "main", "pr") == MAX_BEHIND

    def test_a_clean_merge_without_committer_identity_is_not_a_conflict(
        self, tmp_path, monkeypatch
    ):
        """`git merge --no-ff` without user.name exits 128; merge-tree must not."""
        repo = _init_repo(tmp_path)
        _git(repo, "checkout", "-b", "pr")
        _commit_file(repo, "docs/note.md", "pr\n", "pr commit")
        _git(repo, "checkout", "main")
        _commit_file(repo, "docs/other.md", "main\n", "main commit")
        _git(repo, "config", "--unset", "user.email")
        _git(repo, "config", "--unset", "user.name")
        for key in (
            "GIT_AUTHOR_NAME",
            "GIT_AUTHOR_EMAIL",
            "GIT_COMMITTER_NAME",
            "GIT_COMMITTER_EMAIL",
        ):
            monkeypatch.delenv(key, raising=False)
        verdict = evaluate_refs(repo, "main", "pr", f821_runner=_clean_ruff)
        assert verdict.conclusion is Conclusion.SUCCESS

    def test_a_missing_ref_is_a_git_error_not_a_conflict(self, tmp_path):
        repo = _init_repo(tmp_path)
        verdict = evaluate_refs(repo, "main", "does-not-exist")
        assert verdict.conclusion is Conclusion.FAILURE
        assert "conflict" not in verdict.title


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
    def _loaded(self):
        # PyYAML 1.1 treats `on:` as boolean True.
        data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        assert data is not None
        return data

    def _on(self):
        data = self._loaded()
        return data.get("on", data.get(True))

    def test_the_workflow_file_exists(self):
        assert WORKFLOW.is_file()

    def test_push_to_main_is_a_real_trigger(self):
        """Deleting `on.push` used to leave every grep test green."""
        on = self._on()
        assert on["push"]["branches"] == ["main"]
        assert on["pull_request"]["branches"] == ["main"]
        assert "workflow_dispatch" in on

    def test_it_does_not_use_pull_request_target(self):
        on = self._on()
        assert "pull_request_target" not in on
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "pull_request_target" not in text

    def test_the_reeval_job_checkout_has_no_ref(self):
        """A ref: on this job would run PR code with checks:write."""
        jobs = self._loaded()["jobs"]
        checkout = next(
            step
            for step in jobs["reeval-open-prs"]["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout")
        )
        assert not (checkout.get("with") or {}).get("ref")

    def test_the_pr_job_id_is_not_the_checks_api_name(self):
        jobs = self._loaded()["jobs"]
        assert CHECK_NAME not in jobs
        assert "gate" in jobs
        assert "reeval-open-prs" in jobs

    def test_permissions_can_post_checks_and_cannot_write_contents(self):
        perms = self._loaded()["permissions"]
        assert perms["checks"] == "write"
        assert perms["contents"] == "read"

    def test_the_cap_in_the_workflow_matches_the_script(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        assert f"--max-behind {MAX_BEHIND}" in text

    def test_the_gate_job_fetches_origin_main_as_a_named_ref(self):
        jobs = self._loaded()["jobs"]
        runs = [
            step.get("run", "")
            for step in jobs["gate"]["steps"]
            if "run" in step
        ]
        assert any("main:refs/remotes/origin/main" in run for run in runs)

    def test_concurrency_is_untouched_in_ci_yml(self):
        """#1017's concurrency half is already on main; this PR must not edit it."""
        text = CI.read_text(encoding="utf-8")
        assert "github.event_name == 'pull_request' && github.ref || github.sha" in text

    def test_the_script_uses_merge_tree_not_git_merge(self):
        src = SCRIPT.read_text(encoding="utf-8")
        assert "merge-tree" in src and "--write-tree" in src
        assert "merge --no-edit --no-ff" not in src


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

    def test_the_changelog_fragment_is_named_for_the_pr(self):
        fragment = ROOT / "changelog.d" / "0.75.0" / "1071.fixed.md"
        text = fragment.read_text(encoding="utf-8")
        assert "#1017 by @jagadeepmamidi in #1071" in text
        assert not (ROOT / "changelog.d" / "0.75.0" / "1017.fixed.md").exists()


def _clean_ruff(argv, *, cwd):
    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
