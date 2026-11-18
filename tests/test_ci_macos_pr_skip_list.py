"""The macOS pull-request cell leaves the heaviest platform-neutral test files to the other cells.

2026-11-18: `test (macos-latest, 3.12)` of a pull request took a median of 62.7 minutes and up
to 90 (a 3-worker runner), against 25.6 minutes on Linux, for the same 68,291 tests. In the log
of one green job 26 test files with no platform branch and a minute or more each took 125.6 of
the 262.3 worker-minutes (47.9%), two of them (tests/test_issue379_pincost_main.py and
tests/test_expert_stream_two_read_contract.py) 61, and the same files took 17.4 of the 100.9 on
Linux. The list in `.github/ci/macos-pr-skip.txt` names them; the step "Leave the heavy
platform-neutral test files to the other cells" turns it into `--ignore=` options for pytest,
on the macOS cell of a PULL REQUEST run and nowhere else.

What keeps the rule honest, all of it pinned here:

* a listed file carries no platform or process-model token (`sys.platform`, `darwin`, `mlx`,
  `mps`, `subprocess`, `spawn`, ...): what only macOS can tell stays on macOS;
* the step runs for exactly one kind of cell -- macOS in a pull request. Linux and Windows
  cells, and every cell of the nightly, release, push and manual runs, run the whole suite;
* the same pull-request run has the Linux and Windows cells that run the listed files, and
  nothing else in the workflow, in `pyproject.toml` or in `conftest.py` leaves them out;
* the step reads the list strictly (one test file name per line, nothing else), so a line
  cannot smuggle an option into pytest.

The step's own shell runs here under a local bash, as the runner would run it.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.test_ci_full_matrix_on_approval import (
    ALL_MATRICES,
    BASH,
    NEEDS_BASH,
    ROOT,
    _cell_dicts,
    _jobs,
    _matrix,
)
from tests.test_ci_suite_result_reuse import _contexts, _runs, _steps

LIST = ROOT / ".github" / "ci" / "macos-pr-skip.txt"
STEP = "Leave the heavy platform-neutral test files to the other cells"
RUN_STEPS = ("Run unit tests with coverage", "Run unit tests without coverage")
EVENTS = ("pull_request", "push", "schedule", "workflow_dispatch")
VARIABLE = "PYTEST_ADDOPTS"
#: A list that grows past this hides more than "a few heavy files": raise it on purpose.
MAX_ENTRIES = 40
ENTRY = re.compile(r"^tests/test_[A-Za-z0-9_]+\.py$")

#: What makes a file platform-specific. `\bmps\b` and not `mps`: `json.dumps` is no GPU.
PLATFORM = re.compile(
    r"sys\.platform|platform\.(?:system|machine|processor|mac_ver|uname)|\bdarwin\b|\bwin32\b"
    r"|\bos\.name\b|mlx|\bmps\b|\bapple\b|macos|\bIS_WINDOWS\b|\b_IS_WINDOWS\b",
    re.IGNORECASE,
)
#: What makes a file depend on how the system starts or links processes and files.
PROCESS = re.compile(
    r"subprocess|multiprocessing|spawn|\bfork\b|torch\.distributed|\bgloo\b|signal\.|fcntl"
    r"|mmap|fsync|os\.link|symlink|chmod|st_ino",
    re.IGNORECASE,
)


def entries() -> list[str]:
    """The listed test files, as the step reads them (comments and blank lines dropped)."""
    lines = LIST.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line and not line.startswith("#")]


def _step() -> dict[str, Any]:
    found = [step for step in _steps() if step.get("name") == STEP]
    assert len(found) == 1, f"the test job must have exactly one step named {STEP!r}"
    return found[0]


def _cells(matrix_name: str) -> list[dict[str, str]]:
    return _cell_dicts(_matrix(matrix_name))


def _all_cells() -> list[tuple[str, dict[str, str]]]:
    return [(name, cell) for name in ALL_MATRICES for cell in _cells(name)]


class TestTheScannerSeesWhatItIsMeantTo:
    """The two patterns above are the rule: pin what they catch and what they let through.
    Every sample holds exactly ONE token, so dropping any alternative from the pattern fails one."""

    @pytest.mark.parametrize(
        "text",
        [
            'if sys.platform.startswith("lin"):',
            "name = platform.system()",
            "arch = platform.machine()",
            "cpu = platform.processor()",
            "version = platform.mac_ver()",
            "info = platform.uname()",
            'if kernel == "darwin":',
            'if kernel == "Darwin":',
            'if flavour == "win32":',
            "if os.name == 'nt':",
            "import mlx.core as mx",
            "monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)",
            'pytest.mark.parametrize("device", ["cpu", "mps"])',
            "# Apple silicon only",
            "needs_macos = True",
            "if IS_WINDOWS:",
            "if _IS_WINDOWS:",
        ],
    )
    def test_a_platform_branch_is_seen(self, text: str) -> None:
        assert PLATFORM.search(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "result = subprocess.run(['soup'])",
            "pool = multiprocessing.Pool()",
            "method = 'spawn'",
            "pid = os.fork()",
            "import torch.distributed as dist",
            "backend = 'gloo'",
            "signal.alarm(3)",
            "fcntl.flock(handle, 1)",
            "view = mmap.mmap(handle, 0)",
            "os.fsync(handle)",
            "os.link(first, second)",
            "link = make_symlink(first)",
            "path.chmod(0o600)",
            "inode = info.st_ino",
        ],
    )
    def test_a_process_or_file_system_dependency_is_seen(self, text: str) -> None:
        assert PROCESS.search(text), text

    def test_json_dumps_is_neither(self) -> None:
        text = "tokenizer_config.write_text(json.dumps({'a': 1}))"
        assert not PLATFORM.search(text)
        assert not PROCESS.search(text)


class TestTheListHoldsOnlyFilesWithNothingMacosSpecific:
    def test_it_is_not_empty_and_not_long(self) -> None:
        listed = entries()
        assert listed, "an empty list makes the step a no-op: delete the step instead"
        assert len(listed) <= MAX_ENTRIES, len(listed)

    def test_every_line_is_one_top_level_test_file_name(self) -> None:
        for line in entries():
            assert ENTRY.match(line), repr(line)

    def test_no_entry_twice_and_in_order(self) -> None:
        listed = entries()
        assert listed == sorted(set(listed)), "keep the list sorted, one line per file"

    def test_every_entry_is_a_test_file_that_exists(self) -> None:
        """A renamed file would silently stop being skipped and a deleted one would hide."""
        for line in entries():
            assert (ROOT / line).is_file(), line

    def test_no_entry_has_a_platform_token(self) -> None:
        for line in entries():
            text = (ROOT / line).read_text(encoding="utf-8", errors="replace")
            assert not PLATFORM.findall(text), (line, sorted(set(PLATFORM.findall(text))))

    def test_no_entry_depends_on_how_processes_or_files_are_linked(self) -> None:
        for line in entries():
            text = (ROOT / line).read_text(encoding="utf-8", errors="replace")
            assert not PROCESS.findall(text), (line, sorted(set(PROCESS.findall(text))))

    def test_no_workflow_test_is_listed(self) -> None:
        """The tests that pin this very rule, and every other `test_ci_*`, run on macOS."""
        for line in entries():
            assert not Path(line).name.startswith("test_ci_"), line
        assert "tests/test_ci_macos_pr_skip_list.py" not in entries()

    def test_the_list_file_explains_itself(self) -> None:
        head = [line for line in LIST.read_text(encoding="utf-8").splitlines() if line][:3]
        assert all(line.startswith("#") for line in head), head


class TestTheStepRunsForTheMacosPullRequestCellOnly:
    @pytest.mark.parametrize("event", EVENTS)
    def test_it_runs_exactly_on_the_macos_cell_of_a_pull_request(self, event: str) -> None:
        for matrix_name, cell in _all_cells():
            ran = _runs(_step(), _contexts(cell, event=event, hit="false"))
            expected = event == "pull_request" and cell["os"] == "macos-latest"
            assert ran is expected, (matrix_name, event, cell, ran)

    def test_some_cell_runs_it_and_some_cell_does_not(self) -> None:
        """The two halves above must not both pass for want of cells."""
        ran = [
            _runs(_step(), _contexts(cell, event="pull_request", hit="false"))
            for _, cell in _all_cells()
        ]
        assert any(ran) and not all(ran), ran

    @pytest.mark.parametrize("event", EVENTS)
    def test_a_docs_only_run_does_nothing_here(self, event: str) -> None:
        """Every step of the cells that do no work is skipped by `RUN_SUITE`."""
        for _, cell in _all_cells():
            if cell == {"os": "ubuntu-latest", "python-version": "3.12"}:
                continue
            context = _contexts(cell, event=event, hit="false", docs_only="true")
            assert _runs(_step(), context) is False, (event, cell)

    def test_it_comes_after_the_install_and_before_both_test_steps(self) -> None:
        names = [str(step.get("name") or step.get("uses")) for step in _steps()]
        position = names.index(STEP)
        assert names.index("Install dependencies") < position
        for run in RUN_STEPS:
            assert position < names.index(run), run

    def test_it_carries_the_guards_every_step_of_this_job_carries(self) -> None:
        condition = str(_step()["if"])
        assert "env.RUN_SUITE == 'true'" in condition
        assert "steps.reuse.outputs.cache-hit != 'true'" in condition
        assert "github.event_name == 'pull_request'" in condition
        assert "matrix.os == 'macos-latest'" in condition

    def test_it_is_bounded_and_runs_in_bash(self) -> None:
        step = _step()
        assert step["shell"] == "bash"
        assert step["timeout-minutes"] == 5
        assert "continue-on-error" not in step, "a malformed list must turn the cell red"


class TestEveryListedFileStillRunsWhereTheRuleSaysItDoes:
    def test_the_approval_run_has_the_linux_and_windows_cells_that_run_them(self) -> None:
        cells = _cells("APPROVAL_MATRIX")
        assert {"os": "ubuntu-latest", "python-version": "3.12"} in cells
        assert {"os": "windows-latest", "python-version": "3.12"} in cells
        assert {"os": "macos-latest", "python-version": "3.12"} in cells

    def test_the_release_nightly_and_manual_runs_keep_a_macos_cell_that_runs_all(self) -> None:
        for matrix_name in ("RELEASE_MATRIX", "FULL_MATRIX"):
            assert any(cell["os"] == "macos-latest" for cell in _cells(matrix_name)), matrix_name

    def test_nothing_but_the_step_sets_the_option_variable(self) -> None:
        job = _jobs()["test"]
        assert VARIABLE not in dict(job.get("env") or {})
        for name, other in _jobs().items():
            assert VARIABLE not in dict(other.get("env") or {}), name
            for step in other.get("steps") or []:
                assert VARIABLE not in dict(step.get("env") or {}), (name, step.get("name"))
                if step.get("name") != STEP:
                    assert VARIABLE not in str(step.get("run") or ""), (name, step.get("name"))
                    assert "macos-pr-skip" not in str(step.get("run") or ""), step.get("name")

    @pytest.mark.parametrize("step_name", RUN_STEPS)
    def test_the_test_steps_name_no_file_to_leave_out(self, step_name: str) -> None:
        (step,) = [step for step in _steps() if step.get("name") == step_name]
        command = str(step["run"])
        assert "--ignore" not in command and "--deselect" not in command, command
        assert command.split().count("tests/") == 1, command

    def test_no_entry_is_left_out_by_pyproject_or_conftest(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
        for line in entries():
            name = Path(line).name
            assert name not in pyproject, line
            assert name not in conftest, line

    def test_the_step_is_the_only_place_that_reads_the_list(self) -> None:
        readers = [
            str(step.get("name"))
            for job in _jobs().values()
            for step in job.get("steps") or []
            if "macos-pr-skip" in str(step.get("run") or "")
        ]
        assert readers == [STEP], readers


def _run_step(tmp_path: Path, list_bytes: bytes) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the step's shell in a checkout holding *list_bytes*; return the process and what it
    appended to `$GITHUB_ENV`."""
    assert BASH is not None
    checkout = tmp_path / "checkout"
    (checkout / ".github" / "ci").mkdir(parents=True)
    (checkout / ".github" / "ci" / "macos-pr-skip.txt").write_bytes(list_bytes)
    github_env = tmp_path / "github_env"
    github_env.write_bytes(b"")
    script = tmp_path / "step.sh"
    script.write_bytes(str(_step()["run"]).encode("utf-8"))
    # `shell: bash` is `bash --noprofile --norc -eo pipefail {0}` on the runner.
    result = subprocess.run(
        [BASH, "--noprofile", "--norc", "-eo", "pipefail", script.as_posix()],
        cwd=checkout,
        env={**os.environ, "GITHUB_ENV": github_env.as_posix()},
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    return result, github_env.read_text(encoding="utf-8")


@NEEDS_BASH
class TestTheStepTurnsTheListIntoPytestOptions:
    def test_each_entry_becomes_one_ignore_option_in_order(self, tmp_path: Path) -> None:
        text = b"# a comment\n\ntests/test_a.py\n# another\ntests/test_b_2.py\n"
        result, env = _run_step(tmp_path, text)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert env == "PYTEST_ADDOPTS=--ignore=tests/test_a.py --ignore=tests/test_b_2.py\n"

    def test_a_windows_checkout_with_crlf_line_ends_gives_the_same_options(
        self, tmp_path: Path
    ) -> None:
        result, env = _run_step(tmp_path, b"# c\r\ntests/test_a.py\r\ntests/test_b.py\r\n")
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert env == "PYTEST_ADDOPTS=--ignore=tests/test_a.py --ignore=tests/test_b.py\n"

    def test_a_last_line_without_a_line_end_is_read(self, tmp_path: Path) -> None:
        result, env = _run_step(tmp_path, b"tests/test_a.py")
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert env == "PYTEST_ADDOPTS=--ignore=tests/test_a.py\n"

    def test_the_notice_says_how_many_files_are_left_to_the_other_cells(
        self, tmp_path: Path
    ) -> None:
        result, _ = _run_step(tmp_path, b"tests/test_a.py\ntests/test_b.py\n")
        # ansi-ok: the output of a bash script run in a subprocess, not of a Rich console
        assert "::notice title=macOS pull request cell::2 heavy test files" in result.stdout
        assert "nightly, release and manual runs run them on macOS too" in result.stdout

    @pytest.mark.parametrize(
        "line",
        [
            "tests/test_a.py --ignore=tests/test_b.py",
            "tests/test_a.py -p evil_plugin",
            "tests/test_a.py;touch pwned",
            "tests/../tests/test_a.py",
            "tests/sub/test_a.py",
            "tests/",
            "tests/*.py",
            "tests/test_a.py ",
            "-p evil_plugin",
            "/etc/passwd",
            "tests/conftest.py",
        ],
    )
    def test_a_line_that_is_not_one_test_file_name_fails_the_step(
        self, tmp_path: Path, line: str
    ) -> None:
        result, env = _run_step(tmp_path, f"tests/test_ok.py\n{line}\n".encode())
        assert result.returncode != 0, (line, result.stdout, result.stderr)
        # ansi-ok: the output of a bash script run in a subprocess, not of a Rich console
        assert "::error title=macOS skip list::not a test file name" in result.stdout, line
        assert "pwned" not in env

    def test_the_list_in_the_repository_gives_one_option_per_entry(self, tmp_path: Path) -> None:
        result, env = _run_step(tmp_path, LIST.read_bytes())
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert env.startswith("PYTEST_ADDOPTS=") and env.endswith("\n")
        options = env.removeprefix("PYTEST_ADDOPTS=").split()
        assert options == [f"--ignore={line}" for line in entries()]


class TestContributingSaysWhatTheMacosPullRequestCellLeavesOut:
    def test_it_names_the_list_the_rule_and_who_still_runs_the_files(self) -> None:
        text = " ".join((ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8").split())
        assert "The macOS cell of a pull request leaves out a few heavy test files." in text
        assert ".github/ci/macos-pr-skip.txt" in text
        assert "the Linux and Windows cells of the same run run them" in text
        assert "every cell of the nightly, release and manual runs" in text
        assert "tests/test_ci_macos_pr_skip_list.py" in text
