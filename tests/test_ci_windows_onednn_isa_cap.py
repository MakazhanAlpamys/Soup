"""`ONEDNN_MAX_CPU_ISA` on the Windows cells of the `test` job, and nowhere else.

2026-11-17: on a minority of hosted Windows hosts a pytest-xdist worker dies with `Windows fatal
exception: code 0xc000001d` (illegal instruction) inside torch's CPU bfloat16 `F.linear`, and
xdist then aborts the session at about 46% of the suite. The one host that was printed was an
Intel Xeon Platinum 8573C (a CPU with the AVX-512 bfloat16 and AMX units), though other jobs on
the same model passed. The job-level variable caps oneDNN's instruction set below its bfloat16
and AMX kernels on the Windows cells; every other cell must stay as it was.

The effect on a crashing host cannot be tested here (it needs that host). What can be pinned is
which cells get the cap, that the value is below the bfloat16 / AMX rungs of oneDNN's ladder,
that nothing else in the workflow sets it, and that the step which prints the host prints it.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_ci_full_matrix_on_approval import (
    ALL_MATRICES,
    _cell_dicts,
    _jobs,
    _matrix,
    _render,
)
from tests.test_ci_suite_result_reuse import _contexts, _step, _steps

VARIABLE = "ONEDNN_MAX_CPU_ISA"
CAP = "AVX512_CORE"
EVENTS = ("pull_request", "push", "schedule", "workflow_dispatch")
PRINT_STEP = "Print the host CPU"

# oneDNN's documented values for the variable, lowest first (the "CPU Dispatcher Control" page
# of its developer guide). AVX512_CORE_BF16 is the first rung with the bfloat16 instructions
# and AVX512_CORE_AMX the first with AMX; AVX512_CORE has neither.
LADDER = (
    "AVX2",
    "AVX2_VNNI",
    "AVX512_CORE",
    "AVX512_CORE_VNNI",
    "AVX512_CORE_BF16",
    "AVX512_CORE_FP16",
    "AVX512_CORE_AMX",
)


def _job_env() -> dict[str, Any]:
    return dict(_jobs()["test"]["env"])


def _value_for(cell: dict[str, str], event: str) -> str:
    """The value the runner puts in the job's environment for this cell: the job-level
    expression rendered with the same interpreter the other workflow tests use. A variable the
    job does not define is "" here, which is also what an unset one means to oneDNN."""
    template = _job_env().get(VARIABLE)
    if template is None:
        return ""
    return _render(str(template), _contexts(cell, event=event, hit="false"))


class TestTheCapIsOnTheWindowsCellsAndNowhereElse:
    @pytest.mark.parametrize("matrix_name", ALL_MATRICES)
    @pytest.mark.parametrize("event", EVENTS)
    def test_every_windows_cell_has_the_cap_and_no_other_cell_does(
        self, matrix_name: str, event: str
    ) -> None:
        cells = _cell_dicts(_matrix(matrix_name))
        assert cells, matrix_name
        for cell in cells:
            value = _value_for(cell, event)
            if cell["os"] == "windows-latest":
                assert value == CAP, (matrix_name, event, cell, value)
            else:
                assert value == "", (matrix_name, event, cell, value)

    def test_some_matrix_has_a_windows_cell_and_some_has_a_cell_without(self) -> None:
        """The two halves above must not both pass for want of cells."""
        cells = [
            cell
            for matrix_name in ALL_MATRICES
            for cell in _cell_dicts(_matrix(matrix_name))
        ]
        systems = {cell["os"] for cell in cells}
        assert "windows-latest" in systems, systems
        assert {"ubuntu-latest", "macos-latest"} <= systems, systems

    def test_the_variable_is_set_for_the_whole_job_not_for_one_step(self) -> None:
        """Job level: the suite step, the diagnostic step and any step added later all see
        the same value, and the cap cannot be lost by editing one step."""
        assert VARIABLE in _job_env(), sorted(_job_env())


class TestNothingElseSetsIt:
    def test_no_other_job_and_no_step_sets_it(self) -> None:
        for name, job in _jobs().items():
            if name != "test":
                assert VARIABLE not in dict(job.get("env") or {}), name
            for step in job.get("steps") or []:
                assert VARIABLE not in dict(step.get("env") or {}), (name, step.get("name"))
                # A script may print the variable (the host print does); it may not set it,
                # whether through the environment file or an assignment in its own shell.
                for line in str(step.get("run") or "").splitlines():
                    if VARIABLE in line:
                        assert not any(
                            marker in line
                            for marker in ("GITHUB_ENV", "export ", f"env:{VARIABLE} =", "setx")
                        ), (name, step.get("name"), line)

    def test_the_verbose_switch_is_not_job_wide(self) -> None:
        """ONEDNN_VERBOSE prints a line for every primitive: it belongs to the one command
        that probes the ISA, never to the job that runs the suite."""
        assert "ONEDNN_VERBOSE" not in _job_env(), sorted(_job_env())
        for name, job in _jobs().items():
            for step in job.get("steps") or []:
                assert "ONEDNN_VERBOSE" not in dict(step.get("env") or {}), (
                    name,
                    step.get("name"),
                )


class TestTheValueIsBelowTheBfloat16AndAmxKernels:
    def test_the_cap_is_a_value_onednn_documents(self) -> None:
        assert CAP in LADDER, CAP

    def test_the_cap_is_below_the_first_bfloat16_and_the_first_amx_value(self) -> None:
        rung = LADDER.index(CAP)
        assert rung < LADDER.index("AVX512_CORE_BF16"), CAP
        assert rung < LADDER.index("AVX512_CORE_AMX"), CAP

    def test_the_cap_keeps_avx512_on_hosts_that_have_it(self) -> None:
        """AVX2 would also avoid the bfloat16 / AMX kernels, at a larger cost on every
        AVX-512 host; the value must not be lower than the crash needs."""
        assert LADDER.index(CAP) >= LADDER.index("AVX512_CORE"), CAP

    def test_the_workflow_uses_that_value_and_no_other(self) -> None:
        template = str(_job_env()[VARIABLE])
        assert f"'{CAP}'" in template, template
        for other in LADDER:
            if other != CAP:
                assert f"'{other}'" not in template, (other, template)


class TestTheHostPrintShowsWhetherTheCapWasInForce:
    def _script(self) -> str:
        return str(_step(PRINT_STEP)["run"])

    def test_it_prints_the_value_the_job_gave_the_step(self) -> None:
        assert 'Write-Output "ONEDNN_MAX_CPU_ISA=$env:ONEDNN_MAX_CPU_ISA"' in self._script()

    def test_it_prints_whether_torch_sees_the_bfloat16_and_amx_units(self) -> None:
        script = self._script()
        assert "torch.cpu._is_avx512_bf16_supported()" in script, script
        assert "torch.cpu._is_amx_tile_supported()" in script, script

    def test_it_prints_the_isa_onednn_reports(self) -> None:
        """The probe creates one oneDNN primitive with ONEDNN_VERBOSE on and keeps the line
        that names the ISA, so a host that has the AMX units shows whether the cap hid them."""
        lines = [line.strip() for line in self._script().splitlines()]
        probes = [line for line in lines if "mkldnn_convolution" in line]
        assert len(probes) == 1, lines
        assert 'Select-String -SimpleMatch "info,cpu,isa"' in probes[0], probes[0]
        assert lines.count('$env:ONEDNN_VERBOSE = "1"') == 1, lines
        assert lines.index('$env:ONEDNN_VERBOSE = "1"') == lines.index(probes[0]) - 1, lines

    def test_the_verbose_switch_comes_after_every_other_command_of_the_step(self) -> None:
        lines = [line.strip() for line in self._script().splitlines() if line.strip()]
        assert lines.index('$env:ONEDNN_VERBOSE = "1"') == len(lines) - 2, lines

    def test_the_step_is_still_the_one_diagnostic_that_cannot_fail_the_job(self) -> None:
        step = _step(PRINT_STEP)
        assert step.get("continue-on-error") is True, step
        assert step.get("timeout-minutes") == 5, step
        assert step.get("shell") == "pwsh", step
        assert PRINT_STEP in [str(entry.get("name")) for entry in _steps()]
