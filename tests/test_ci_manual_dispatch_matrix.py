"""A manual dispatch of CI can ask for the approval matrix instead of the nightly one.

Measured 2026-11-12..15 (72 hours): nine manual dispatches of the full matrix on one
integration branch cost 3,404 job-minutes, 15% of everything the repository ran, and a tenth
cost 396 on a feature branch. A FULL run is seven test cells on the newest upstream stack; the
cells a pull request labelled ``ci:full`` runs are four, on the pinned stack. What FULL adds is
the three Python 3.13 cells and the nightly-only jobs: about 95 job-minutes of a run that costs
215 when nothing is re-run.

So ``workflow_dispatch`` takes one input, ``matrix``: ``full`` (the default, and what every
dispatch was until now) or ``approval``. With ``approval`` the ``plan`` job picks the APPROVAL
matrix and the pinned stack -- exactly what a ``ci:full`` pull request gets -- and the jobs that
need ``full`` start as they do for any run that is not a pull request.

Nothing else reads the input. The nightly run, a push, a release branch and a pull request see
no ``inputs`` at all, and a dispatch that names nothing is still FULL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_ci_full_matrix_on_approval import (
    BADGE_STEP,
    CONTRIBUTING,
    COVERAGE_CELL,
    COVERAGE_STEP,
    LABEL,
    NEEDS_BASH,
    NO_COVERAGE_STEP,
    PINNED_CONSTRAINTS,
    UPLOAD_STEP,
    _approval_for,
    _cell_dicts,
    _cells_running,
    _decide,
    _full_for,
    _group,
    _jobs,
    _literal,
    _matrix,
    _on_event,
    _pull_request,
    _push,
    _release_for,
    _runs,
    _suite_steps_in,
    _test_step,
    _triggers,
    _workflow,
)


def _dispatch(matrix: str | None) -> dict[str, Any]:
    """A dispatch context; GitHub fills the default, so ``None`` means no ``inputs`` at all."""
    return _on_event("workflow_dispatch", inputs=None if matrix is None else {"matrix": matrix})


def _with_inputs(context: dict[str, Any], matrix: str) -> dict[str, Any]:
    return {**context, "inputs": {"matrix": matrix}}


class TestTheInput:
    def test_dispatch_declares_one_choice_input_that_defaults_to_full(self) -> None:
        inputs = _triggers()["workflow_dispatch"]["inputs"]
        assert list(inputs) == ["matrix"]
        matrix = inputs["matrix"]
        assert matrix["type"] == "choice"
        assert matrix["options"] == ["full", "approval"]
        assert matrix["default"] == "full"
        assert "approval" in str(matrix["description"]) and "full" in str(matrix["description"])

    def test_no_other_trigger_takes_inputs(self) -> None:
        triggers = _triggers()
        for name, body in triggers.items():
            if name != "workflow_dispatch" and isinstance(body, dict):
                assert "inputs" not in body, name


class TestTheDecision:
    @pytest.mark.parametrize("matrix", ("full", None))
    def test_a_dispatch_for_full_or_for_nothing_is_the_nightly_matrix(
        self, matrix: str | None
    ) -> None:
        context = _dispatch(matrix)
        assert _full_for(context) == "true"
        assert _approval_for(context) == "false"
        assert _release_for(context) == "false"

    def test_a_dispatch_for_approval_is_the_approval_matrix(self) -> None:
        context = _dispatch("approval")
        assert _full_for(context) == "false"
        assert _approval_for(context) == "true"
        assert _release_for(context) == "false"

    @pytest.mark.parametrize("matrix", ("approval", "full", "APPROVAL", "x"))
    def test_only_the_word_approval_selects_it(self, matrix: str) -> None:
        """The comparison is case-insensitive, as GitHub's is; anything else stays FULL."""
        context = _dispatch(matrix)
        approval = matrix.lower() == "approval"
        assert _approval_for(context) == ("true" if approval else "false")
        assert _full_for(context) == ("false" if approval else "true")

    @pytest.mark.parametrize(
        "context",
        [
            pytest.param(_on_event("schedule"), id="nightly"),
            pytest.param(_push("refs/heads/main"), id="push-main"),
            pytest.param(_push("refs/heads/release/v0.76.0"), id="push-release"),
            pytest.param(_pull_request("opened"), id="pr-opened"),
            pytest.param(_pull_request("synchronize", labels=(LABEL,)), id="pr-ci-full"),
        ],
    )
    def test_no_other_event_reads_the_input(self, context: dict[str, Any]) -> None:
        """The same decision whether an ``inputs.matrix`` of either value is present or not."""
        before = (_full_for(context), _approval_for(context), _release_for(context))
        for matrix in ("approval", "full"):
            after = _with_inputs(context, matrix)
            assert (_full_for(after), _approval_for(after), _release_for(after)) == before

    def test_the_label_still_makes_a_pull_request_an_approval_run_only_by_itself(self) -> None:
        assert _approval_for(_pull_request("labeled", label=LABEL)) == "true"
        assert _approval_for(_pull_request("labeled", label="bug")) == "false"

    def test_the_two_choices_never_select_both(self) -> None:
        for matrix in (None, "full", "approval"):
            context = _dispatch(matrix)
            assert not (_full_for(context) == "true" and _approval_for(context) == "true")


@NEEDS_BASH
class TestWhatThePlanStepOutputs:
    """The step's own shell, run under bash with the env its expressions render to."""

    def _outputs(self, tmp_path: Path, matrix: str | None) -> dict[str, str]:
        context = _dispatch(matrix)
        return _decide(
            tmp_path,
            release=_release_for(context),
            full=_full_for(context),
            approval=_approval_for(context),
        )

    def test_approval_gets_the_approval_matrix_and_the_pinned_stack(self, tmp_path: Path) -> None:
        outputs = self._outputs(tmp_path, "approval")
        assert outputs["full"] == "true"
        assert json.loads(outputs["matrix"]) == json.loads(_literal("APPROVAL_MATRIX"))
        assert outputs["constraints"] == f"-c {PINNED_CONSTRAINTS}"

    @pytest.mark.parametrize("matrix", ("full", None))
    def test_full_keeps_the_nightly_matrix_and_the_newest_stack(
        self, tmp_path: Path, matrix: str | None
    ) -> None:
        outputs = self._outputs(tmp_path, matrix)
        assert outputs["full"] == "true"
        assert json.loads(outputs["matrix"]) == json.loads(_literal("FULL_MATRIX"))
        assert outputs["constraints"] == ""


class TestWhatStillRunsWithTheApprovalMatrix:
    def _starts(self, job: str, matrix: str | None) -> bool:
        context = _dispatch(matrix)
        full = _full_for(context) == "true" or _approval_for(context) == "true"
        context = {**context, "needs": {"plan": {"outputs": {"full": "true" if full else "false"}}}}
        return _runs(job, context)

    @pytest.mark.parametrize(
        "job",
        (
            "mlx-smoke",
            "pytorch-smoke",
            "transformers-floor",
            "core-floor",
            "type-check",
            "torchao-contract",
            "typer-vendored",
        ),
    )
    @pytest.mark.parametrize("matrix", ("approval", "full"))
    def test_every_job_a_dispatch_had_still_starts(self, job: str, matrix: str) -> None:
        """The three required smokes and the floor job, and the ones a pull request skips."""
        assert self._starts(job, matrix), (job, matrix)


class TestCoverageInAnApprovalDispatch:
    """A dispatch is not a pull request, so it measures coverage like every run that is not one:
    ubuntu / 3.11 (a cell the approval matrix has) measures, enforces the floor and uploads;
    every other cell runs the suite with `--no-cov`; the badge is the nightly run's alone."""

    def test_only_ubuntu_3_11_measures_and_uploads(self) -> None:
        context = _dispatch("approval")
        assert _cells_running(_test_step(COVERAGE_STEP), "APPROVAL_MATRIX", context) == [
            COVERAGE_CELL
        ]
        assert _cells_running(_test_step(UPLOAD_STEP), "APPROVAL_MATRIX", context) == [
            COVERAGE_CELL
        ]

    def test_every_cell_runs_the_suite_exactly_once(self) -> None:
        context = _dispatch("approval")
        for cell in _cell_dicts(_matrix("APPROVAL_MATRIX")):
            expected = COVERAGE_STEP if cell == COVERAGE_CELL else NO_COVERAGE_STEP
            assert _suite_steps_in(cell, context) == [expected], cell

    def test_the_badge_is_not_updated(self) -> None:
        context = _dispatch("approval")
        assert _cells_running(_test_step(BADGE_STEP), "APPROVAL_MATRIX", context) == []


class TestNothingElseMoved:
    def test_a_dispatch_is_still_alone_in_a_group_of_its_own_commit(self) -> None:
        context = _dispatch("approval")
        assert _group(context) == f"CI-workflow_dispatch-{context['github']['sha']}"
        assert _group(context) == _group(_dispatch("full"))

    def test_cancellation_is_still_pull_request_only(self) -> None:
        assert _workflow()["concurrency"]["cancel-in-progress"] == (
            "${{ github.event_name == 'pull_request' }}"
        )

    def test_the_required_test_job_is_not_conditional(self) -> None:
        assert "if" not in _jobs()["test"]


def _contributing() -> str:
    return " ".join(CONTRIBUTING.read_text(encoding="utf-8").split())


def test_contributing_tells_a_maintainer_how_to_ask_for_the_approval_matrix() -> None:
    text = _contributing()
    assert "`gh workflow run ci.yml --ref <branch> -f matrix=approval`" in text
    assert "The default is still the nightly matrix" in text
