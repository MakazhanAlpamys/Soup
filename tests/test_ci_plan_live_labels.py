"""`plan` reads the pull request's labels when it RUNS, not from the event that started the run.

Measured 2026-11-21..22 (timelines of #3156 and #3166, runs 44886605066/44886605569 and
45015709674/45015710513): a maintainer adds ``ci:full`` a second or two after a push. GitHub
makes two runs, one for the ``labeled`` event and one for the push, and both share the pull
request's concurrency group, so the LATER-CREATED run cancels the earlier one. The push event
was generated before the label existed, so its payload lists no ``ci:full``; when its run is
created after the ``labeled`` run, it wins the group and computes the QUICK matrix. The pull
request is left with ubuntu / 3.12 green and the other required contexts never reported, and
the label has to be removed and added again (11 of the last 100 merged pull requests carry
exactly that remove-and-re-add pair on their timeline).

The fix is one step in ``plan`` that asks the API for the pull request's CURRENT labels, so
whichever run survives the group computes the matrix from the same truth. The event's payload
is only the fallback when the API cannot be read. Nothing here talks to GitHub: the step's own
shell runs under a local bash with ``gh`` replaced by a function, and the expressions are
evaluated by the interpreter of tests/test_ci_full_matrix_on_approval.py.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.test_ci_full_matrix_on_approval import (
    BASH,
    EVENTS,
    LABEL,
    NEEDS_BASH,
    ROOT,
    _approval_for,
    _decide,
    _decide_step,
    _full_for,
    _group,
    _matrix,
    _plan,
    _pull_request,
    _release_for,
)

#: The filter the step hands `gh api --jq`: prints the word `true` or `false`. The fake `gh`
#: below prints what that filter would print; a change to the filter must change this line.
JQ_FILTER = 'any(.labels[]; (.name | ascii_downcase) == "ci:full")'
#: The pull-request events that start a run (`unlabeled` is not a trigger).
PULL_REQUEST_ACTIONS = ("opened", "synchronize", "reopened", "ready_for_review", "labeled")


def _labels_step() -> dict[str, Any]:
    found = [step for step in _plan()["steps"] if step.get("id") == "labels"]
    assert len(found) == 1, "the plan job must have exactly one step with `id: labels`"
    return found[0]


def _payload_has_label(context: dict[str, Any]) -> bool:
    labels = context["github"]["event"]["pull_request"]["labels"]
    return any(item["name"].lower() == LABEL for item in labels)


def _with_live_label(context: dict[str, Any], present: bool) -> dict[str, Any]:
    """*context* as `_approval_for` reads it when the labels step found *present* live."""
    return {
        **context,
        "steps": {"labels": {"outputs": {"ci_full": "true" if present else "false"}}},
    }


def _predating_the_label(action: str) -> dict[str, Any]:
    """An event of *action* whose payload was generated before ci:full was added to the pull
    request: no `ci:full` in its labels (for `labeled`, the event of ANOTHER label)."""
    context = _pull_request(action, label="bug" if action == "labeled" else None)
    assert not _payload_has_label(context)
    return context


class _Fake:
    """The result of running the `labels` step's script against a canned API answer."""

    def __init__(self, output: list[str], gh_arguments: list[str] | None, log: str) -> None:
        self.output = output
        self.gh_arguments = gh_arguments
        self.log = log

    @property
    def ci_full(self) -> str:
        assert len(self.output) == 1 and self.output[0].startswith("ci_full="), self.output
        return self.output[0].partition("=")[2]


def _run_labels_step(
    tmp_path: Path,
    *,
    live: str = "",
    payload: str = "false",
    event: str = "pull_request",
    number: str = "7",
    gh_status: int = 0,
) -> _Fake:
    assert BASH is not None
    answer = tmp_path / "answer.txt"
    answer.write_bytes(f"{live}\n".encode() if live != "" else b"")
    log = tmp_path / "gh_arguments"
    output = tmp_path / "github_output"
    output.write_bytes(b"")
    stub = tmp_path / "gh_stub.sh"
    stub.write_bytes(
        b"gh() {\n"
        b'  printf "%s\\n" "$@" > "$FAKE_GH_LOG"\n'
        b'  if [ "$FAKE_GH_STATUS" != "0" ]; then\n'
        b'    echo "gh: HTTP 502" >&2\n'
        b'    return "$FAKE_GH_STATUS"\n'
        b"  fi\n"
        b'  cat "$FAKE_GH_ANSWER"\n'
        b"}\n"
    )
    script = tmp_path / "labels.sh"
    script.write_bytes(str(_labels_step()["run"]).encode("utf-8"))
    env = {
        **os.environ,
        "BASH_ENV": stub.as_posix(),
        "GH_TOKEN": "not-a-token",
        "EVENT_NAME": event,
        "REPOSITORY": "MakazhanAlpamys/Soup",
        # Set even for an event that has no pull request: only the event may start the call.
        "PR_NUMBER": number,
        "EVENT_HAS_LABEL": payload,
        "GITHUB_OUTPUT": output.as_posix(),
        "FAKE_GH_ANSWER": answer.as_posix(),
        "FAKE_GH_LOG": log.as_posix(),
        "FAKE_GH_STATUS": str(gh_status),
    }
    # The runner's default shell for a step without `shell:` is `bash -e {0}`.
    result = subprocess.run(
        [BASH, "--noprofile", "--norc", "-e", script.as_posix()],
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    arguments = log.read_text(encoding="utf-8").splitlines() if log.exists() else None
    return _Fake(output.read_text(encoding="utf-8").splitlines(), arguments, result.stdout)


# --- Where the step sits and what it is given -------------------------------------------


class TestTheLabelsStepIsWiredIntoPlan:
    def test_it_runs_before_the_decision_that_reads_it(self) -> None:
        ids = [step.get("id") for step in _plan()["steps"]]
        assert ids.index("labels") < ids.index("decide")

    def test_it_is_given_its_inputs_through_env_and_never_through_the_script(self) -> None:
        env = _labels_step()["env"]
        assert env == {
            "GH_TOKEN": "${{ github.token }}",
            "EVENT_NAME": "${{ github.event_name }}",
            "REPOSITORY": "${{ github.repository }}",
            "PR_NUMBER": "${{ github.event.pull_request.number }}",
            "EVENT_HAS_LABEL": (
                "${{ contains(github.event.pull_request.labels.*.name, 'ci:full') }}"
            ),
        }
        assert "${{" not in str(_labels_step()["run"])

    def test_it_reads_the_one_pull_request_and_nothing_else(self) -> None:
        script = str(_labels_step()["run"])
        assert 'gh api "repos/${REPOSITORY}/pulls/${PR_NUMBER}"' in script
        assert f"--jq '{JQ_FILTER}'" in script
        for other in ("--paginate", "git ", "issues/", "--method", "-X ", "--input", "-f ", "-F "):
            assert other not in script, other

    def test_plan_still_holds_no_permission_but_reading_pull_requests(self) -> None:
        """`GET /pulls/{n}` needs `pull-requests: read`, which `plan` already has."""
        assert _plan()["permissions"] == {"pull-requests": "read"}

    def test_the_decision_reads_the_step_and_no_longer_the_event_payload(self) -> None:
        expression = str(_decide_step()["env"]["APPROVAL"])
        assert "steps.labels.outputs.ci_full == 'true'" in expression
        assert "labels.*.name" not in expression, "the payload is the fallback, inside the step"
        assert "github.event.label" not in expression


# --- What the step computes -------------------------------------------------------------


@NEEDS_BASH
class TestTheLabelsStepAnswersFromTheLiveLabels:
    def test_a_label_added_after_the_event_was_generated_is_seen(self, tmp_path: Path) -> None:
        """#3156 and #3166: the push event has no ci:full in its payload, the pull request has
        it by the time the run's plan job executes."""
        assert _run_labels_step(tmp_path, live="true", payload="false").ci_full == "true"

    def test_a_label_removed_while_the_run_waited_for_a_runner_is_not_used(
        self, tmp_path: Path
    ) -> None:
        """`plan` waits a median of 30 minutes for a runner on an approval run: the maintainer's
        current choice is the one that counts, as the label's documentation says."""
        assert _run_labels_step(tmp_path, live="false", payload="true").ci_full == "false"

    @pytest.mark.parametrize(("live", "payload"), [("true", "true"), ("false", "false")])
    def test_agreement_is_passed_through(self, tmp_path: Path, live: str, payload: str) -> None:
        assert _run_labels_step(tmp_path, live=live, payload=payload).ci_full == live

    def test_the_call_is_this_pull_requests_own_endpoint(self, tmp_path: Path) -> None:
        fake = _run_labels_step(tmp_path, live="true", number="1234")
        assert fake.gh_arguments == [
            "api",
            "repos/MakazhanAlpamys/Soup/pulls/1234",
            "--jq",
            JQ_FILTER,
        ]

    @pytest.mark.parametrize("payload", ("true", "false"))
    def test_an_api_error_falls_back_to_the_event_and_says_so(
        self, tmp_path: Path, payload: str
    ) -> None:
        """What the workflow did before this step existed, and a warning on the run."""
        fake = _run_labels_step(tmp_path, payload=payload, gh_status=1)
        assert fake.ci_full == payload
        assert "::warning title=Label check::could not read" in fake.log, fake.log

    @pytest.mark.parametrize("answer", ("yes", "null", "True", "[]", "1"))
    @pytest.mark.parametrize("payload", ("true", "false"))
    def test_an_answer_that_is_not_the_word_true_or_false_falls_back(
        self, tmp_path: Path, answer: str, payload: str
    ) -> None:
        fake = _run_labels_step(tmp_path, live=answer, payload=payload)
        assert fake.ci_full == payload, answer
        assert "::warning title=Label check::unexpected answer" in fake.log, fake.log

    def test_an_empty_answer_falls_back(self, tmp_path: Path) -> None:
        fake = _run_labels_step(tmp_path, live="", payload="true")
        assert fake.ci_full == "true"
        assert "::warning title=Label check::unexpected answer" in fake.log, fake.log

    @pytest.mark.parametrize("event", ("push", "schedule", "workflow_dispatch"))
    def test_a_run_that_is_not_a_pull_request_never_has_the_label(
        self, tmp_path: Path, event: str
    ) -> None:
        """No API call, and the matrix of these runs is decided by the event alone."""
        fake = _run_labels_step(tmp_path, live="true", payload="true", event=event)
        assert fake.ci_full == "false"
        assert fake.gh_arguments is None, "gh was called outside a pull request"

    def test_a_pull_request_without_a_number_uses_the_event_without_a_call(
        self, tmp_path: Path
    ) -> None:
        fake = _run_labels_step(tmp_path, live="true", payload="false", number="")
        assert fake.ci_full == "false"
        assert fake.gh_arguments is None
        assert _run_labels_step(tmp_path, live="false", payload="true", number="").ci_full == "true"

    def test_only_the_exact_word_true_in_the_payload_counts(self, tmp_path: Path) -> None:
        fake = _run_labels_step(tmp_path, payload="yes", gh_status=1)
        assert fake.ci_full == "false"

    def test_exactly_one_line_is_written_to_the_output(self, tmp_path: Path) -> None:
        for live in ("true", "false"):
            assert _run_labels_step(tmp_path, live=live).output == [f"ci_full={live}"]


# --- What the matrix does with it -------------------------------------------------------


class TestEveryEventGetsTheMatrixOfTheLiveLabel:
    @pytest.mark.parametrize("action", PULL_REQUEST_ACTIONS)
    def test_a_push_whose_payload_predates_the_label_still_gets_the_full_matrix(
        self, action: str
    ) -> None:
        """The inverted race: whichever event's run is created last, and whatever its payload
        says, a pull request that carries ci:full when `plan` runs gets APPROVAL."""
        stale = _predating_the_label(action)
        assert not _payload_has_label(stale)
        assert _approval_for(_with_live_label(stale, True)) == "true", action

    @pytest.mark.parametrize("action", PULL_REQUEST_ACTIONS)
    def test_a_label_removed_since_the_event_gets_the_quick_matrix(self, action: str) -> None:
        label = LABEL if action == "labeled" else None
        context = _pull_request(action, label=label, labels=(LABEL,))
        assert _payload_has_label(context)
        assert _approval_for(_with_live_label(context, False)) == "false", action

    def test_the_two_runs_of_the_race_agree_whichever_one_survives_the_group(self) -> None:
        """A `labeled` run and a push run of one head share a group (the later one cancels the
        earlier), and after the fix they compute the same matrix, so which one is later does
        not matter."""
        labeled = _pull_request("labeled", label=LABEL, run_id="1")
        pushed = _pull_request("synchronize", run_id="2")
        assert _group(labeled) == _group(pushed)
        assert _payload_has_label(labeled) and not _payload_has_label(pushed)
        assert _approval_for(_with_live_label(labeled, True)) == "true"
        assert _approval_for(_with_live_label(pushed, True)) == "true"

    @pytest.mark.parametrize(("context", "kind"), EVENTS)
    def test_a_default_context_still_means_what_the_payload_says(self, context, kind) -> None:
        """`_approval_for` fills in the step's answer from the payload when a test gives none:
        the table in tests/test_ci_full_matrix_on_approval.py keeps its meaning."""
        assert _approval_for(context) == ("true" if kind == "APPROVAL" else "false")

    def test_non_pull_request_events_ignore_the_step_output(self) -> None:
        for context, expected in (
            (
                {
                    "github": {"event_name": "push", "ref": "refs/heads/main"},
                    "steps": {"labels": {"outputs": {"ci_full": "true"}}},
                },
                "false",
            ),
            (
                {
                    "github": {"event_name": "workflow_dispatch"},
                    "inputs": {"matrix": "approval"},
                    "steps": {"labels": {"outputs": {"ci_full": "false"}}},
                },
                "true",
            ),
            (
                {
                    "github": {"event_name": "workflow_dispatch"},
                    "inputs": {"matrix": "full"},
                    "steps": {"labels": {"outputs": {"ci_full": "true"}}},
                },
                "false",
            ),
        ):
            assert _approval_for(context) == expected

    @NEEDS_BASH
    @pytest.mark.parametrize("action", PULL_REQUEST_ACTIONS)
    def test_end_to_end_the_decide_step_picks_approval_for_a_label_the_payload_missed(
        self, tmp_path: Path, action: str
    ) -> None:
        """The events' expressions, the fake API answer and the decide step's own shell."""
        stale = _with_live_label(_predating_the_label(action), True)
        outputs = _decide(
            tmp_path,
            release=_release_for(stale),
            full=_full_for(stale),
            approval=_approval_for(stale),
        )
        assert outputs["full"] == "true"
        assert json.loads(outputs["matrix"]) == _matrix("APPROVAL_MATRIX")


class TestContributingSaysWhereTheLabelIsRead:
    def test_it_explains_the_live_read_and_event_fallback(self) -> None:
        text = " ".join((ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8").split())
        assert "The plan job reads the current pull request labels when it runs" in text
        assert "the job warns and falls back to the labels in its triggering event" in text
        assert (
            "If the required cells are still missing after that fallback, "
            "a maintainer removes and re-adds `ci:full`"
        ) in text
