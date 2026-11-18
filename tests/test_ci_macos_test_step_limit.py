"""The macOS cells of the `test` job have a longer time limit than the other cells.

2026-11-18, commit 14be6a450 (release/v0.78.0, 68,291 tests): `test (macos-latest, 3.12)` was
stopped twice by the step limit of 90 minutes -- "The action 'Run unit tests without
coverage' has timed out after 90 minutes", at 98% of the run, with no failed test (run
44320568935, attempts 1 and 2) -- and a third run of the same commit passed with the step at
89.8 minutes (run 44320582523). The runner gave pytest-xdist three workers (`-n logical`), and a
macOS step is 10 minutes on a runner that gives it five, so the same suite is a coin flip
against 90 on the slow ones.

The limit is raised on the macOS cells only. Windows and Linux finished far inside 90 minutes
on the same commit and keep their limits, so a hang there is still ended at once. What is
pinned here, per matrix, per event and per cell, rendered with the interpreter of
tests/test_ci_full_matrix_on_approval.py: the step limit of both test steps, the job limit
that must stay above it, and that the evidence the number rests on has the margin it claims.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_ci_full_matrix_on_approval import (
    ALL_MATRICES,
    CONTRIBUTING,
    _cell_dicts,
    _jobs,
    _matrix,
    _render,
)
from tests.test_ci_suite_result_reuse import _contexts, _steps

RUN_STEPS = ("Run unit tests with coverage", "Run unit tests without coverage")
EVENTS = ("pull_request", "push", "schedule", "workflow_dispatch")

#: Every other cell: what the whole `test` job had until 2026-11-18.
DEFAULT_STEP_MINUTES = 90
DEFAULT_JOB_MINUTES = 105
#: The macOS cells.
MACOS_STEP_MINUTES = 120
MACOS_JOB_MINUTES = 135
#: Checkout, install and the cache warm-up come before the suite step, and the job has to
#: outlast them (they took under 2 minutes on the macOS runner of run 44320582523).
JOB_MARGIN_MINUTES = 15

#: Measured on 2026-11-18 (see the module docstring): the slowest macOS step that PASSED.
SLOWEST_GREEN_MACOS_STEP_MINUTES = 89.8
#: What it takes to finish the run that the old limit cut off at 98%: 90 / 0.98.
CUT_OFF_RUN_MINUTES = 92
HEADROOM = 1.3


def minutes_in(raw: Any, cell: dict[str, str], *, event: str = "pull_request") -> int:
    """The limit GitHub applies to *cell*: a plain number as written, an expression rendered
    with the same interpreter the other workflow tests use."""
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    rendered = _render(str(raw), _contexts(cell, event=event, hit="false"))
    assert rendered.isdigit(), (raw, rendered)
    return int(rendered)


def _step_template(name: str) -> Any:
    (step,) = [step for step in _steps() if step.get("name") == name]
    assert "timeout-minutes" in step, name
    return step["timeout-minutes"]


def _job_template() -> Any:
    return _jobs()["test"]["timeout-minutes"]


def _all_cells() -> list[tuple[str, dict[str, str]]]:
    return [
        (matrix_name, cell)
        for matrix_name in ALL_MATRICES
        for cell in _cell_dicts(_matrix(matrix_name))
    ]


def _expected(cell: dict[str, str]) -> tuple[int, int]:
    if cell["os"] == "macos-latest":
        return MACOS_STEP_MINUTES, MACOS_JOB_MINUTES
    return DEFAULT_STEP_MINUTES, DEFAULT_JOB_MINUTES


class TestOnlyTheMacosCellsGetTheLongerLimit:
    @pytest.mark.parametrize("event", EVENTS)
    @pytest.mark.parametrize("step_name", RUN_STEPS)
    def test_the_step_limit_of_every_cell(self, step_name: str, event: str) -> None:
        for matrix_name, cell in _all_cells():
            got = minutes_in(_step_template(step_name), cell, event=event)
            assert got == _expected(cell)[0], (matrix_name, event, step_name, cell, got)

    @pytest.mark.parametrize("event", EVENTS)
    def test_the_job_limit_of_every_cell(self, event: str) -> None:
        for matrix_name, cell in _all_cells():
            got = minutes_in(_job_template(), cell, event=event)
            assert got == _expected(cell)[1], (matrix_name, event, cell, got)

    def test_the_two_test_steps_share_one_limit(self) -> None:
        """The coverage step runs in one ubuntu cell and the other in every other cell: a
        limit written differently in the two would move with the step that happens to run."""
        for _, cell in _all_cells():
            assert minutes_in(_step_template(RUN_STEPS[0]), cell) == minutes_in(
                _step_template(RUN_STEPS[1]), cell
            ), cell

    def test_the_matrices_have_macos_cells_and_cells_without(self) -> None:
        """The two halves above must not both pass for want of cells."""
        systems = {cell["os"] for _, cell in _all_cells()}
        assert {"macos-latest", "ubuntu-latest", "windows-latest"} <= systems, systems

    def test_macos_is_the_only_system_whose_limit_differs(self) -> None:
        limits = {
            cell["os"]: minutes_in(_step_template(RUN_STEPS[1]), cell) for _, cell in _all_cells()
        }
        assert limits["ubuntu-latest"] == limits["windows-latest"] == DEFAULT_STEP_MINUTES
        assert limits["macos-latest"] > DEFAULT_STEP_MINUTES, limits

    def test_a_cell_that_is_not_macos_keeps_the_old_numbers_in_the_text(self) -> None:
        """The previous limits stay the written default, so reverting the macOS branch is a
        one-token change and a reader finds both numbers where the limit is set."""
        for template in (
            _step_template(RUN_STEPS[0]),
            _step_template(RUN_STEPS[1]),
            _job_template(),
        ):
            text = str(template)
            assert "matrix.os" in text and "'macos-latest'" in text, text
        assert "|| 90 }}" in str(_step_template(RUN_STEPS[1]))
        assert "|| 105 }}" in str(_job_template())


class TestTheJobLimitStaysAboveTheStepLimit:
    def test_the_job_outlasts_the_step_by_the_setup_margin_in_every_cell(self) -> None:
        for matrix_name, cell in _all_cells():
            job = minutes_in(_job_template(), cell)
            for step_name in RUN_STEPS:
                step = minutes_in(_step_template(step_name), cell)
                assert job >= step + JOB_MARGIN_MINUTES, (matrix_name, cell, step_name, job, step)

    def test_the_margin_the_pinned_numbers_have_is_the_margin_this_file_demands(self) -> None:
        assert MACOS_JOB_MINUTES - MACOS_STEP_MINUTES >= JOB_MARGIN_MINUTES
        assert DEFAULT_JOB_MINUTES - DEFAULT_STEP_MINUTES >= JOB_MARGIN_MINUTES


class TestTheMacosNumberRestsOnTheEvidence:
    def test_it_is_above_the_slowest_passing_step_with_the_headroom_it_claims(self) -> None:
        assert MACOS_STEP_MINUTES >= HEADROOM * SLOWEST_GREEN_MACOS_STEP_MINUTES

    def test_it_is_above_the_run_the_old_limit_cut_off(self) -> None:
        assert MACOS_STEP_MINUTES > CUT_OFF_RUN_MINUTES

    def test_it_stays_close_enough_to_the_evidence_not_to_hide_a_hang(self) -> None:
        """A step that is stopped must be stopped soon: the limit is not allowed to creep to
        twice what the slowest healthy run took."""
        assert MACOS_STEP_MINUTES <= 1.5 * SLOWEST_GREEN_MACOS_STEP_MINUTES

    def test_the_old_limit_was_inside_the_evidence(self) -> None:
        """Why this file exists: 90 is below what a healthy run needed on the slow runners."""
        assert DEFAULT_STEP_MINUTES < CUT_OFF_RUN_MINUTES
        assert DEFAULT_STEP_MINUTES - SLOWEST_GREEN_MACOS_STEP_MINUTES < 1

    def test_the_per_test_cap_that_catches_a_real_hang_was_not_touched(self) -> None:
        for name in RUN_STEPS:
            (step,) = [step for step in _steps() if step.get("name") == name]
            run = str(step["run"])
            assert "--timeout=600 --timeout-method=thread --max-worker-restart=2" in run, name


class TestTheExpressionEvaluatorReadsNumbers:
    """The limits are numbers inside an expression; the evaluator the workflow tests share
    used to refuse a number literal."""

    def test_a_number_branch_is_taken_for_the_matching_system(self) -> None:
        template = "${{ matrix.os == 'macos-latest' && 120 || 90 }}"
        assert _render(template, {"matrix": {"os": "macos-latest"}}) == "120"
        assert _render(template, {"matrix": {"os": "ubuntu-latest"}}) == "90"
        assert _render(template, {"matrix": {"os": "windows-latest"}}) == "90"

    def test_a_zero_branch_falls_through_like_github_does(self) -> None:
        template = "${{ matrix.os == 'macos-latest' && 0 || 90 }}"
        assert _render(template, {"matrix": {"os": "macos-latest"}}) == "90"


class TestContributingNamesTheLimits:
    def _text(self) -> str:
        return " ".join(CONTRIBUTING.read_text(encoding="utf-8").split())

    def test_the_default_and_the_macos_limits_are_both_written_down(self) -> None:
        text = self._text()
        assert "a 90-minute step limit and a 105-minute job limit" in text
        macos = f"{MACOS_STEP_MINUTES} and {MACOS_JOB_MINUTES} minutes on the macOS cells"
        assert macos in text, "CONTRIBUTING.md must name the macOS limits"
