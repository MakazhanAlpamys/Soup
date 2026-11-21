"""The `Print the host CPU` step of the `test` job: Windows only, and it cannot fail the job.

2026-11-16: on a minority of Windows hosts two xdist workers died with `Windows fatal
exception: code 0xc000001d` (illegal instruction) inside torch's CPU bfloat16 `F.linear`, and
pytest-xdist 3.8.0 then aborted the session at about 46% of the suite. The log never named the
CPU, so the cause could not be narrowed to a host class. The step prints it.

It is a diagnostic: a Windows cell that is red because this step failed would be a worse
failure than the one it explains, so it must never fail the job, never run where it has no
use, and never pull in code from outside the repository.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_ci_full_matrix_on_approval import ALL_MATRICES, _cell_dicts, _matrix
from tests.test_ci_suite_result_reuse import (
    COVERAGE_SUITE,
    DIAGNOSTICS,
    SUITE,
    _contexts,
    _names,
    _runs,
    _step,
)

NAME = "Print the host CPU"
INSTALL = "Install dependencies"


def test_the_reuse_tests_know_this_step_as_their_one_diagnostic() -> None:
    """tests/test_ci_suite_result_reuse.py lets exactly this step carry `continue-on-error`
    besides the reuse machinery; the two files must name the same step."""
    assert DIAGNOSTICS == (NAME,), DIAGNOSTICS


def _the_step() -> dict[str, Any]:
    return _step(NAME)


class TestTheStepNeverFailsTheJob:
    def test_a_failure_of_the_step_is_not_a_failure_of_the_job(self) -> None:
        """`continue-on-error: true` is what keeps a missing Get-CimInstance or a torch that
        will not import from turning a cell red for a reason unrelated to the tests."""
        assert _the_step().get("continue-on-error") is True, _the_step()

    def test_a_hung_step_cannot_hold_the_runner(self) -> None:
        """Bounded like the reuse machinery: a Get-CimInstance that never returns must not
        spend the job's 105 minutes."""
        assert _the_step().get("timeout-minutes") == 5, _the_step()

    def test_it_uses_no_action_and_no_secret(self) -> None:
        step = _the_step()
        assert "uses" not in step, step
        assert "secrets." not in str(step), step
        assert "GITHUB_TOKEN" not in str(step), step


class TestWhereTheStepRuns:
    def test_it_sits_after_the_install_and_before_every_test_step(self) -> None:
        names = _names()
        assert names.count(NAME) == 1, names
        assert names.index(INSTALL) < names.index(NAME) < names.index(COVERAGE_SUITE), names
        assert names.index(NAME) < names.index(SUITE), names

    @pytest.mark.parametrize("matrix_name", ALL_MATRICES)
    @pytest.mark.parametrize("event", ["pull_request", "push", "schedule", "workflow_dispatch"])
    def test_it_runs_on_the_windows_cells_and_nowhere_else(
        self, matrix_name: str, event: str
    ) -> None:
        step = _the_step()
        cells = _cell_dicts(_matrix(matrix_name))
        assert cells, matrix_name
        for cell in cells:
            context = _contexts(cell, event=event, hit="false")
            if _runs(step, context):
                assert cell["os"] == "windows-latest", (matrix_name, event, cell)

    def test_every_matrix_that_has_a_windows_cell_runs_it_there(self) -> None:
        """The other half of the rule above: Windows-only must not become never."""
        step = _the_step()
        windows = [
            (matrix_name, cell)
            for matrix_name in ALL_MATRICES
            for cell in _cell_dicts(_matrix(matrix_name))
            if cell["os"] == "windows-latest"
        ]
        assert windows, "no matrix has a Windows cell"
        for matrix_name, cell in windows:
            context = _contexts(cell, event="schedule", hit="false")
            assert _runs(step, context), (matrix_name, cell)

    def test_it_does_not_run_where_the_suite_is_reused_or_skipped(self) -> None:
        step = _the_step()
        windows = {"os": "windows-latest", "python-version": "3.12"}
        assert not _runs(step, _contexts(windows, event="pull_request", hit="true"))
        assert not _runs(
            step, _contexts(windows, event="pull_request", hit="false", docs_only="true")
        )


class TestWhatTheStepPrints:
    def test_it_prints_the_cpu_identity_and_what_torch_will_dispatch_to(self) -> None:
        step = _the_step()
        assert step.get("shell") == "pwsh", step
        script = str(step["run"])
        for needle in (
            "PROCESSOR_IDENTIFIER",
            "Get-CimInstance Win32_Processor",
            "NumberOfCores",
            "NumberOfLogicalProcessors",
            "torch.backends.cpu.get_cpu_capability()",
        ):
            assert needle in script, (needle, script)
