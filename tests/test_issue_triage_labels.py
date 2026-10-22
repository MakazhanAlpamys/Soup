"""The issue-triage label rules: `needs-triage` on new issues, one `difficulty:N` per issue.

The rules live in `scripts/issue_triage_labels.py` and run from
`.github/workflows/issue-triage.yml` on `issues` events. The workflow reads only the issue
number, the repository and the event action; it never reads issue text.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import issue_triage_labels as triage
from scripts.issue_triage_labels import NEEDS_TRIAGE, Decision, decide

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "issue-triage.yml"

LEVELS = [f"difficulty:{level}" for level in range(1, 11)]


class TestDecide:
    def test_a_new_issue_with_no_level_gets_needs_triage(self) -> None:
        assert decide([], "opened") == Decision(add=(NEEDS_TRIAGE,))

    def test_the_form_type_label_does_not_stop_it(self) -> None:
        assert decide(["bug"], "opened") == Decision(add=(NEEDS_TRIAGE,))

    @pytest.mark.parametrize("label", LEVELS)
    def test_an_issue_filed_with_a_level_is_left_alone(self, label: str) -> None:
        assert decide(["bug", label], "opened") == Decision()

    def test_opening_is_idempotent(self) -> None:
        assert decide(["needs-triage"], "opened") == Decision()

    @pytest.mark.parametrize("label", LEVELS)
    def test_a_level_removes_needs_triage(self, label: str) -> None:
        assert decide([NEEDS_TRIAGE, label], "labeled") == Decision(remove=(NEEDS_TRIAGE,))

    def test_a_level_filed_together_with_needs_triage_removes_it_on_open(self) -> None:
        assert decide([NEEDS_TRIAGE, "difficulty:3"], "opened") == Decision(
            remove=(NEEDS_TRIAGE,)
        )

    def test_needs_triage_alone_stays_until_a_level_is_set(self) -> None:
        assert decide([NEEDS_TRIAGE, "bug"], "labeled") == Decision()

    def test_removing_a_level_does_not_bring_needs_triage_back(self) -> None:
        assert decide(["bug"], "unlabeled") == Decision()

    def test_labelling_a_later_event_never_adds_needs_triage(self) -> None:
        assert decide([], "labeled") == Decision()

    def test_it_only_ever_touches_needs_triage(self) -> None:
        labels = ["bug", "claimed", "help wanted", "ci:full", "difficulty:5", NEEDS_TRIAGE]
        decision = decide(labels, "labeled")
        assert set(decision.add) | set(decision.remove) <= {NEEDS_TRIAGE}


class TestTheOneLevelGuard:
    def test_two_levels_are_an_error_naming_both(self) -> None:
        decision = decide(["difficulty:7", "difficulty:3"], "labeled")
        assert "difficulty:3" in decision.error
        assert "difficulty:7" in decision.error

    def test_the_levels_are_listed_in_a_stable_order(self) -> None:
        first = decide(["difficulty:7", "difficulty:3"], "labeled").error
        second = decide(["difficulty:3", "difficulty:7"], "labeled").error
        assert first == second

    def test_ten_sorts_as_a_number_not_as_text(self) -> None:
        assert decide(["difficulty:10", "difficulty:2"], "labeled").error.index(
            "difficulty:2"
        ) < decide(["difficulty:10", "difficulty:2"], "labeled").error.index("difficulty:10")

    @pytest.mark.parametrize(
        "label", ["difficulty:0", "difficulty:11", "difficulty:x", "difficulty:", "difficulty:05"]
    )
    def test_an_unknown_level_is_an_error(self, label: str) -> None:
        decision = decide([label], "labeled")
        assert "unknown difficulty label" in decision.error
        assert label in decision.error

    def test_one_valid_level_is_no_error(self) -> None:
        assert decide(["difficulty:4"], "labeled").error == ""

    def test_two_levels_still_clear_needs_triage(self) -> None:
        decision = decide([NEEDS_TRIAGE, "difficulty:2", "difficulty:3"], "labeled")
        assert decision.remove == (NEEDS_TRIAGE,)
        assert decision.error

    def test_a_label_that_only_looks_similar_is_not_a_level(self) -> None:
        assert decide(["not-difficulty:3", "difficulty-3"], "labeled") == Decision()


class FakeGh:
    """Stands in for `subprocess.run(["gh", ...])` and records every call.

    `labels` is what the first `gh issue view` returns; each entry of `later` is what the next
    view returns (the last one repeats), so a test can change the issue between two reads.
    """

    def __init__(self, labels: list[str], later: list[list[str]] | None = None) -> None:
        self.states = [labels, *(later or [])]
        self.views = 0
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        self.calls.append(command)
        if command[:3] == ["gh", "issue", "view"]:
            names = self.states[min(self.views, len(self.states) - 1)]
            self.views += 1
            stdout = json.dumps({"labels": [{"name": name} for name in names]})
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    @property
    def edits(self) -> list[list[str]]:
        return [call for call in self.calls if call[:3] == ["gh", "issue", "edit"]]


def _env(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    base = {"REPO": "MakazhanAlpamys/Soup", "NUMBER": "1700", "EVENT_ACTION": "opened"}
    for key in ("REPO", "NUMBER", "EVENT_ACTION"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {**base, **values}.items():
        monkeypatch.setenv(key, value)


class TestMain:
    def test_a_new_issue_is_edited_once_with_exactly_the_right_arguments(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake = FakeGh(["bug"])
        _env(monkeypatch)
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 0
        assert fake.edits == [
            [
                "gh", "issue", "edit", "1700", "--repo", "MakazhanAlpamys/Soup",
                "--add-label", "needs-triage",
            ]
        ]
        assert "needs-triage" in capsys.readouterr().out

    def test_it_reads_the_current_labels_not_the_event_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeGh([])
        _env(monkeypatch)
        monkeypatch.setattr(triage.subprocess, "run", fake)
        triage.main()
        assert fake.calls[0][:3] == ["gh", "issue", "view"]
        assert "labels" in fake.calls[0]

    def test_a_level_removes_needs_triage_through_the_cli(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeGh(["needs-triage", "difficulty:4"])
        _env(monkeypatch, EVENT_ACTION="labeled")
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 0
        assert fake.edits[0][-2:] == ["--remove-label", "needs-triage"]

    def test_nothing_to_do_means_no_edit_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeGh(["difficulty:4"])
        _env(monkeypatch, EVENT_ACTION="labeled")
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 0
        assert fake.edits == []

    def test_two_levels_exit_nonzero_with_an_annotation_but_still_fix_what_it_can(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake = FakeGh(["needs-triage", "difficulty:2", "difficulty:3"])
        _env(monkeypatch, EVENT_ACTION="labeled")
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 1
        assert "::error::" in capsys.readouterr().out
        assert fake.edits and fake.edits[0][-2:] == ["--remove-label", "needs-triage"]

    @pytest.mark.parametrize("number", ["", "abc", "12 34", "1700; rm -rf /", "-1", "1700\n"])
    def test_a_bad_issue_number_never_reaches_gh(
        self, monkeypatch: pytest.MonkeyPatch, number: str
    ) -> None:
        fake = FakeGh([])
        _env(monkeypatch, NUMBER=number)
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 2
        assert fake.calls == []

    @pytest.mark.parametrize(
        "repo",
        [
            "", "Soup", "a/b/c", "owner/rep o", "--repo=x/y", "o/r;id", "../..", "a/..", "./x",
            "x/.", "..", "a/b\n", "-a/b", "a/-b",
        ],
    )
    def test_a_bad_repository_never_reaches_gh(
        self, monkeypatch: pytest.MonkeyPatch, repo: str
    ) -> None:
        fake = FakeGh([])
        _env(monkeypatch, REPO=repo)
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 2
        assert fake.calls == []

    def test_a_missing_variable_is_a_clear_exit_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake = FakeGh([])
        _env(monkeypatch)
        monkeypatch.delenv("NUMBER")
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 2
        assert "NUMBER" in capsys.readouterr().out
        assert fake.calls == []

    def test_a_failing_label_write_is_reported_even_when_the_read_worked(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reader = FakeGh([])

        def read_ok_write_fails(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess:
            if command[:3] == ["gh", "issue", "edit"]:
                return subprocess.CompletedProcess(
                    command, 1, stdout="", stderr="could not add label: 'needs-triage' not found"
                )
            return reader(command, **kwargs)

        _env(monkeypatch)
        monkeypatch.setattr(triage.subprocess, "run", read_ok_write_fails)
        assert triage.main() == 1
        output = capsys.readouterr().out
        assert "::error::" in output
        assert "gh issue edit" in output
        assert "not found" in output

    def test_a_failing_gh_call_is_reported_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def broken(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="HTTP 403: nope")

        _env(monkeypatch)
        monkeypatch.setattr(triage.subprocess, "run", broken)
        assert triage.main() == 1
        assert "403" in capsys.readouterr().out


class TestMoreOfMain:
    @pytest.mark.parametrize("repo", ["MakazhanAlpamys/Soup", "a/.github", "a.b/c-d_e", "A1/b2"])
    def test_ordinary_repository_names_are_accepted(
        self, monkeypatch: pytest.MonkeyPatch, repo: str
    ) -> None:
        fake = FakeGh(["difficulty:2"])
        _env(monkeypatch, REPO=repo, EVENT_ACTION="labeled")
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 0
        first = fake.calls[0]
        assert first[first.index("--repo") + 1] == repo

    @pytest.mark.parametrize("action", ["", "edited", "closed", "OPENED", "opened\n", "labeled "])
    def test_an_unknown_event_action_is_refused_before_any_gh_call(
        self, monkeypatch: pytest.MonkeyPatch, action: str
    ) -> None:
        fake = FakeGh([])
        _env(monkeypatch, EVENT_ACTION=action)
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 2
        assert fake.calls == []

    @pytest.mark.parametrize("action", sorted(triage.ACTIONS))
    def test_every_action_the_workflow_listens_for_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, action: str
    ) -> None:
        fake = FakeGh(["difficulty:2"])
        _env(monkeypatch, EVENT_ACTION=action)
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 0

    def test_the_actions_match_the_workflow_triggers(self, workflow: dict) -> None:
        assert set(workflow[True]["issues"]["types"]) == set(triage.ACTIONS)

    def test_a_level_set_while_the_opened_run_was_working_clears_needs_triage_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeGh(["bug"], later=[["bug", "difficulty:3", "needs-triage"]])
        _env(monkeypatch)
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 0
        assert [edit[-2:] for edit in fake.edits] == [
            ["--add-label", "needs-triage"],
            ["--remove-label", "needs-triage"],
        ]

    def test_without_a_late_level_the_opened_run_edits_once_and_reads_twice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeGh(["bug"], later=[["bug", "needs-triage"]])
        _env(monkeypatch)
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 0
        assert len(fake.edits) == 1
        assert fake.views == 2

    def test_a_labels_run_that_adds_nothing_does_not_read_twice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeGh(["needs-triage", "difficulty:3"])
        _env(monkeypatch, EVENT_ACTION="labeled")
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 0
        assert fake.views == 1

    def test_a_failing_second_read_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reads = {"count": 0}

        def second_read_fails(
            command: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess:
            if command[:3] == ["gh", "issue", "view"]:
                reads["count"] += 1
                if reads["count"] == 2:
                    return subprocess.CompletedProcess(command, 1, stdout="", stderr="HTTP 502")
                return subprocess.CompletedProcess(
                    command, 0, stdout='{"labels": []}', stderr=""
                )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        _env(monkeypatch)
        monkeypatch.setattr(triage.subprocess, "run", second_read_fails)
        assert triage.main() == 1
        assert "502" in capsys.readouterr().out

    def test_a_label_name_cannot_start_a_second_workflow_command(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        nasty = "difficulty:7\n::add-mask::secret\r::error::forged"
        fake = FakeGh([nasty, "difficulty:3"])
        _env(monkeypatch, EVENT_ACTION="labeled")
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 1
        lines = capsys.readouterr().out.splitlines()
        commands = [line for line in lines if line.startswith("::")]
        assert len(commands) == 1 and commands[0].startswith("::error::")
        assert "%0A" in commands[0] and "%0D" in commands[0]
        assert "\r" not in commands[0]

    def test_a_percent_sign_in_a_label_name_is_escaped(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake = FakeGh(["difficulty:5%0A::add-mask::x"])
        _env(monkeypatch, EVENT_ACTION="labeled")
        monkeypatch.setattr(triage.subprocess, "run", fake)
        assert triage.main() == 1
        output = capsys.readouterr().out
        assert "%250A" in output
        assert [line for line in output.splitlines() if line.startswith("::")] == [
            line for line in output.splitlines() if line.startswith("::error::")
        ]

    def test_gh_error_text_with_a_line_break_cannot_start_a_command(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def broken(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr="HTTP 500%\r::add-mask::x"
            )

        _env(monkeypatch)
        monkeypatch.setattr(triage.subprocess, "run", broken)
        assert triage.main() == 1
        lines = capsys.readouterr().out.splitlines()
        assert [line for line in lines if line.startswith("::")] == [
            line for line in lines if line.startswith("::error::")
        ]
        assert len(lines) == 1


@pytest.fixture(scope="module")
def text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow(text: str) -> dict:
    return yaml.safe_load(text)


class TestTheWorkflowFile:
    def test_it_runs_on_issue_events_and_nothing_else(self, workflow: dict) -> None:
        triggers = workflow[True]  # YAML 1.1 reads the key `on` as the boolean True
        assert set(triggers) == {"issues"}
        assert set(triggers["issues"]["types"]) == {"opened", "labeled", "unlabeled"}

    def test_the_token_can_only_write_issues_and_read_contents(self, workflow: dict) -> None:
        assert workflow["permissions"] == {"issues": "write", "contents": "read"}

    def test_the_only_third_party_step_is_checkout(self, text: str) -> None:
        uses = re.findall(r"^\s*(?:-\s*)?uses:\s*(\S+)", text, flags=re.MULTILINE)
        assert uses and all(item.startswith("actions/checkout@") for item in uses)

    def test_no_issue_text_is_ever_read(self, text: str) -> None:
        for forbidden in ("issue.title", "issue.body", "comment.body", "head_ref"):
            assert forbidden not in text, forbidden

    def test_no_expression_is_interpolated_into_a_shell_command(self, workflow: dict) -> None:
        steps = [step for job in workflow["jobs"].values() for step in job["steps"]]
        for step in steps:
            assert "${{" not in step.get("run", ""), step

    def test_the_values_reach_the_script_as_environment_variables(self, workflow: dict) -> None:
        steps = [step for job in workflow["jobs"].values() for step in job["steps"]]
        env = next(step["env"] for step in steps if "issue_triage_labels" in step.get("run", ""))
        assert {"GH_TOKEN", "REPO", "NUMBER", "EVENT_ACTION"} <= set(env)

    def test_it_is_bounded(self, workflow: dict) -> None:
        job = next(iter(workflow["jobs"].values()))
        assert job["timeout-minutes"] <= 10

    def test_no_job_widens_the_permissions(self, workflow: dict) -> None:
        for name, job in workflow["jobs"].items():
            assert "permissions" not in job, name

    def test_checkout_is_the_default_branch_with_no_token_left_behind(
        self, workflow: dict
    ) -> None:
        steps = [step for job in workflow["jobs"].values() for step in job["steps"]]
        checkouts = [step for step in steps if str(step.get("uses", "")).startswith("actions/")]
        assert len(checkouts) == 1
        assert checkouts[0]["with"] == {"persist-credentials": False}
        assert "ref" not in checkouts[0]["with"]

    def test_the_concurrency_is_per_issue_on_the_job_and_never_cancels(
        self, workflow: dict
    ) -> None:
        assert "concurrency" not in workflow
        job = next(iter(workflow["jobs"].values()))
        assert "github.event.issue.number" in job["concurrency"]["group"]
        assert job["concurrency"]["cancel-in-progress"] is False

    def test_the_opened_run_has_a_group_no_label_event_can_share(self, workflow: dict) -> None:
        group = next(iter(workflow["jobs"].values()))["concurrency"]["group"]
        assert "github.event.action == 'opened' && 'opened' || 'labels'" in group

    def test_only_the_events_that_can_change_the_outcome_take_a_runner(
        self, workflow: dict
    ) -> None:
        condition = " ".join(next(iter(workflow["jobs"].values()))["if"].split())
        assert condition == (
            "github.event.action == 'opened' || "
            "startsWith(github.event.label.name, 'difficulty:') || "
            "github.event.label.name == 'needs-triage'"
        )

    def test_the_label_prefix_in_the_condition_matches_the_script(self, workflow: dict) -> None:
        condition = next(iter(workflow["jobs"].values()))["if"]
        assert "'difficulty:'" in condition
        assert f"'{NEEDS_TRIAGE}'" in condition
