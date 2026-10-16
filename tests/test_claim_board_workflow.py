"""The daily claim-board workflow (.github/workflows/claim-board.yml).

It runs with a write token on a schedule, so its shape is pinned here: what triggers it, what
the token may do, which action it runs and how, and that no expression reaches a shell.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "claim-board.yml"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"


@pytest.fixture(scope="module")
def text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow(text: str) -> dict:
    return yaml.safe_load(text)


@pytest.fixture(scope="module")
def steps(workflow: dict) -> list[dict]:
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


def test_it_runs_daily_or_by_hand_and_on_nothing_else(workflow: dict) -> None:
    triggers = workflow[True]  # YAML 1.1 reads the key `on` as the boolean True
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert triggers["schedule"] == [{"cron": "17 5 * * *"}]


def test_the_token_reads_code_and_writes_only_issues(workflow: dict) -> None:
    assert workflow["permissions"] == {
        "contents": "read",
        "issues": "write",
        "pull-requests": "read",
    }
    for name, job in workflow["jobs"].items():
        assert "permissions" not in job, name


def test_the_workflow_uses_the_workflow_token_and_no_secret(text: str, steps: list[dict]) -> None:
    step = next(step for step in steps if "claim_board.py" in step.get("run", ""))
    assert step["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert not re.search(r"\bsecrets\s*(?:\.|\[)", text)


def test_it_never_runs_in_a_fork(workflow: dict) -> None:
    for name, job in workflow["jobs"].items():
        assert job["if"] == "github.repository == 'MakazhanAlpamys/Soup'", name


def test_the_only_action_is_checkout_pinned_by_commit(text: str, steps: list[dict]) -> None:
    uses = re.findall(r"^\s*(?:-\s*)?uses:\s*(\S+)", text, flags=re.MULTILINE)
    assert len(uses) == 1
    assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", uses[0]), uses[0]
    checkout = next(step for step in steps if "uses" in step)
    assert checkout["with"] == {"persist-credentials": False}


def test_no_expression_is_interpolated_into_a_shell_command(steps: list[dict]) -> None:
    runs = [step["run"] for step in steps if "run" in step]
    assert runs
    for run in runs:
        assert "${{" not in run, run


def test_no_issue_text_is_read_by_the_workflow_itself(text: str) -> None:
    for forbidden in ("issue.title", "issue.body", "comment.body", "head_ref", "event.inputs"):
        assert forbidden not in text, forbidden


def test_runs_never_overlap_or_cancel_each_other(workflow: dict) -> None:
    for name, job in workflow["jobs"].items():
        assert job["concurrency"] == {"group": "claim-board", "cancel-in-progress": False}, name
        assert job["timeout-minutes"] <= 10, name


def test_it_rewrites_the_issue_contributing_points_to(steps: list[dict]) -> None:
    step = next(step for step in steps if "claim_board.py" in step.get("run", ""))
    board = step["env"]["BOARD_ISSUE"]
    assert board.isdigit()
    assert 'gh issue edit "$BOARD_ISSUE"' in step["run"]
    assert "--body-file claim-board.md" in step["run"]
    assert "--write claim-board.md" in step["run"]
    assert f"https://github.com/MakazhanAlpamys/Soup/issues/{board}" in CONTRIBUTING.read_text(
        encoding="utf-8"
    )
