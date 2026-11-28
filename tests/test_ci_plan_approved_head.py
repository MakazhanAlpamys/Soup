"""Exercise the approval head probe from ci.yml with a stub GitHub CLI."""

import os
import subprocess
from pathlib import Path

import pytest

from tests.test_ci_full_matrix_on_approval import (
    BASH,
    LABEL,
    NEEDS_BASH,
    _approval_for,
    _Expression,
    _plan,
    _pull_request,
)

HEAD = "b" * 40
LIVE_HEAD_FILTER = 'if .state == "open" and .draft == false then .head.sha else "" end'


def _probe():
    return next(step for step in _plan()["steps"] if step.get("id") == "approval")


def test_probe_only_runs_on_the_explicit_label_event():
    step = _probe()
    assert step["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "REPOSITORY": "${{ github.repository }}",
        "PR_NUMBER": "${{ github.event.pull_request.number }}",
        "EVENT_HEAD": "${{ github.event.pull_request.head.sha }}",
    }
    assert "${{" not in step["run"]
    for action in ("opened", "ready_for_review", "reopened", "synchronize", "labeled"):
        for label in (LABEL, LABEL.upper(), "docs"):
            event = _pull_request(action, labels=(LABEL,), label=label)
            expected = action == "labeled" and label.lower() == LABEL
            assert _Expression(step["if"], event).value() is expected


def _run_probe(tmp_path: Path, *, live: str, status: int, event_head: str):
    assert BASH is not None
    script = tmp_path / "probe.sh"
    script.write_bytes(_probe()["run"].encode())
    stub = tmp_path / "gh_stub.sh"
    stub.write_bytes(
        b"gh() {\n"
        b'  printf "%s\\n" "$@" > "$FAKE_GH_ARGUMENTS"\n'
        b'  printf "%s\\n" "$FAKE_GH_HEAD"\n'
        b'  return "$FAKE_GH_STATUS"\n'
        b"}\n"
    )
    output = tmp_path / "output"
    args = tmp_path / "args"
    env = {
        **os.environ,
        "BASH_ENV": stub.as_posix(),
        "REPOSITORY": "MakazhanAlpamys/Soup",
        "PR_NUMBER": "7",
        "EVENT_HEAD": event_head,
        "FAKE_GH_HEAD": live,
        "FAKE_GH_STATUS": str(status),
        "FAKE_GH_ARGUMENTS": args.as_posix(),
        "GITHUB_OUTPUT": output.as_posix(),
    }
    result = subprocess.run(
        [BASH, "--noprofile", "--norc", "-e", script.as_posix()],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert args.read_text(encoding="utf8").splitlines() == [
        "api",
        "repos/MakazhanAlpamys/Soup/pulls/7",
        "--jq",
        LIVE_HEAD_FILTER,
    ]
    return output.read_text(encoding="utf8").strip(), result.stdout


@NEEDS_BASH
@pytest.mark.parametrize(
    ("live", "status", "event_head", "current"),
    [
        (HEAD, 0, HEAD, "true"),
        ("c" * 40, 0, HEAD, "false"),
        ("", 0, HEAD, "false"),
        (HEAD, 1, HEAD, "false"),
        ("unexpected", 0, "unexpected", "false"),
    ],
)
def test_approval_probe_requires_the_exact_live_open_head(
    tmp_path,
    live,
    status,
    event_head,
    current,
):
    output, log = _run_probe(tmp_path, live=live, status=status, event_head=event_head)
    assert output == f"current={current}"
    assert ("::warning" in log) is bool(status)
    event = _pull_request("labeled", label=LABEL)
    event["steps"] = {"approval": {"outputs": {"current": current}}}
    assert _approval_for(event) == current
