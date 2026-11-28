"""Pin exact-head approval routing and the stale-run cancellation trust boundary."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.test_ci_full_matrix_on_approval import (
    LABEL,
    NEEDS_BASH,
    _approval_for,
    _decide,
    _group,
    _jobs,
    _pull_request,
    _render,
    _workflow,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / ".github/workflows/ci-cancel-obsolete-approval.yml"
SMOKES = ("mlx-smoke", "pytorch-smoke", "transformers-floor")


@pytest.mark.parametrize("action", ["opened", "synchronize", "reopened", "ready_for_review"])
def test_label_presence_alone_does_not_approve(action):
    event = _pull_request(action, labels=(LABEL,))
    assert _approval_for(event) == "false"
    for name in SMOKES:
        assert _render(_jobs()[name]["name"], event) != name


def test_explicit_approval_has_an_independent_group_and_required_names():
    event = _pull_request("labeled", label=LABEL)
    assert _approval_for(event) == "true"
    assert _group(event) != _group(_pull_request("synchronize", labels=(LABEL,)))
    assert _render(_workflow()["run-name"], event) == "PR #7 approval"
    context = {**event, "needs": {"plan": {"outputs": {"full": "true"}}}}
    for name in SMOKES:
        assert _render(_jobs()[name]["name"], context) == name


def test_unrelated_label_cannot_request_approval_or_replace_required_lint():
    event = _pull_request("labeled", labels=(LABEL,), label="docs")
    assert _approval_for(event) == "false"
    assert _render(_jobs()["lint"]["name"], event) != "lint"


def test_stale_run_controller_is_default_branch_only_and_never_checks_out_code():
    workflow = yaml.load(CONTROL.read_text(encoding="utf8"), Loader=yaml.BaseLoader)
    assert workflow["on"] == {
        "pull_request_target": {"branches": ["main"], "types": ["synchronize"]}
    }
    assert workflow["permissions"] == {}
    assert len(workflow["jobs"]) == 1
    job = workflow["jobs"]["supersede"]
    assert job["permissions"] == {"actions": "write", "pull-requests": "read"}
    assert job["runs-on"] == "ubuntu-latest"
    assert len(job["steps"]) == 1
    step = job["steps"][0]
    assert "uses" not in step and "${{" not in step["run"]
    assert step["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "REPOSITORY": "${{ github.repository }}",
        "PR_NUMBER": "${{ github.event.pull_request.number }}",
    }


LIVE_HEAD = "b" * 40
STALE_HEAD = "a" * 40
SOURCE = 321
BRANCH = "work/answer"
REPO = "MakazhanAlpamys/Soup"


def _fork_run(run_id, **overrides):
    return {
        "id": run_id,
        "path": ".github/workflows/ci.yml",
        "event": "pull_request",
        "display_title": "PR #7 approval",
        "head_sha": STALE_HEAD,
        "head_branch": BRANCH,
        "head_repository": {"id": SOURCE},
        "pull_requests": [],
        "status": "in_progress",
        **overrides,
    }


def _execute_controller(monkeypatch, runs, *, live_heads=(LIVE_HEAD, LIVE_HEAD)):
    workflow = yaml.load(CONTROL.read_text(encoding="utf8"), Loader=yaml.BaseLoader)
    shell = workflow["jobs"]["supersede"]["steps"][0]["run"]
    source = shell.split("\n", 1)[1].removesuffix("PY\n")
    results = iter(live_heads)
    called = []
    monkeypatch.setenv("REPOSITORY", REPO)
    monkeypatch.setenv("PR_NUMBER", "7")

    def run(argv, **kwargs):
        called.append(argv)
        assert argv[:2] == ["gh", "api"]
        url = argv[2]
        if url == f"repos/{REPO}/pulls/7":
            response = {
                "number": 7,
                "state": "open",
                "base": {"repo": {"full_name": REPO}, "ref": "main"},
                "head": {"repo": {"id": SOURCE}, "ref": BRANCH, "sha": next(results)},
            }
        elif url == (
            f"repos/{REPO}/actions/workflows/ci.yml/runs?"
            "event=pull_request&branch=work%2Fanswer&per_page=100"
        ):
            assert argv[3:] == ["--paginate", "--slurp"]
            response = [{"workflow_runs": runs}]
        elif url == f"repos/{REPO}/actions/runs/101/cancel":
            assert argv[3:] == ["--method", "POST", "--silent"]
            response = {}
        else:
            pytest.fail(f"unexpected gh argument: {argv!r}")
        return SimpleNamespace(stdout=json.dumps(response))

    monkeypatch.setattr(subprocess, "run", run)
    exec(compile(source, str(CONTROL), "exec"), {})
    return [argv[2] for argv in called if "--method" in argv]


def test_supersession_targets_only_an_old_approval_for_this_pr(monkeypatch):
    runs = [
        _fork_run(101, status="queued"),
        _fork_run(102, head_sha=LIVE_HEAD),
        _fork_run(103, head_repository={"id": SOURCE + 1}),
        _fork_run(104, display_title="PR #8 approval"),
        _fork_run(105, path=".github/workflows/some-other.yml"),
        _fork_run(106, status="completed"),
        _fork_run(107, head_branch="other"),
        _fork_run(108, pull_requests=[{"number": 8}]),
        _fork_run(109, head_sha="invalid"),
        _fork_run(110, event="push"),
    ]
    assert _execute_controller(monkeypatch, runs) == [f"repos/{REPO}/actions/runs/101/cancel"]


def test_head_drift_after_listing_cancels_nothing(monkeypatch):
    assert (
        _execute_controller(monkeypatch, [_fork_run(101)], live_heads=(LIVE_HEAD, STALE_HEAD)) == []
    )


def test_approval_groups_are_bound_to_the_exact_pr_head():
    earlier = _pull_request("labeled", label=LABEL)
    later = _pull_request("labeled", label=LABEL)
    earlier["github"]["event"]["pull_request"]["head"] = {"sha": "a" * 40}
    later["github"]["event"]["pull_request"]["head"] = {"sha": "b" * 40}
    assert _group(earlier) != _group(later)


@NEEDS_BASH
def test_quick_summary_explains_how_to_approve_the_current_head(tmp_path, monkeypatch):
    summary = tmp_path / "summary"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", summary.as_posix())
    _decide(tmp_path, release="false", full="false", approval="false")
    text = summary.read_text(encoding="utf-8")
    assert "CI mode is **Quick**" in text
    assert "remove and re-add `ci:full` on the current head" in text
