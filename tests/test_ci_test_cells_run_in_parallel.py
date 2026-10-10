"""The `test` cells of CI run the suite in parallel, one test file per worker.

Measured on 2026-10-07 with Python 3.12 (run 37577416935 of the pytest-xdist trial, #1380 and
#1672): ubuntu 2520 s -> 1133 s, Windows 4492 s -> 1390 s, macOS 2003 s -> 992 s, with the same
tests passing and coverage unchanged. These pins keep the three choices that made that run
clean:

* `--dist loadfile`, never `load`: the tests of one file stay in one process and in order;
* `-n logical` with the plugin's `psutil` extra installed, which is what counts the CPUs;
* nothing in `pyproject.toml` turns workers on, so a local `pytest tests/` stays serial.

pyproject is read with regular expressions, not ``tomllib``, which is 3.11+ while this
repository supports 3.10: the convention of ``tests/test_requires_python_bound.py``.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = ROOT / "pyproject.toml"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"

RUN_STEP = "Run unit tests with coverage"
INSTALL_STEP = "Install dependencies"


def _jobs() -> dict[str, Any]:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]


def _step(name: str) -> dict[str, Any]:
    steps = [step for step in _jobs()["test"]["steps"] if step.get("name") == name]
    assert len(steps) == 1, (name, steps)
    return steps[0]


def _pytest_args() -> list[str]:
    args = shlex.split(str(_step(RUN_STEP)["run"]))
    assert args[:2] == ["pytest", "tests/"], args
    return args


def _option(args: list[str], flag: str) -> list[str]:
    """Every value given to ``flag``, as ``flag value`` or ``flag=value``."""
    values = [args[index + 1] for index, arg in enumerate(args[:-1]) if arg == flag]
    values += [arg.split("=", 1)[1] for arg in args if arg.startswith(flag + "=")]
    return values


class TestTheTestJobRunsInParallel:
    def test_one_worker_per_logical_cpu(self) -> None:
        assert _option(_pytest_args(), "-n") == ["logical"]

    def test_one_test_file_per_worker(self) -> None:
        """`load` hands the tests of one file to several workers: module-level state and
        test order inside a file are lost, and a timing test failed under it."""
        assert _option(_pytest_args(), "--dist") == ["loadfile"]

    def test_the_plugin_is_installed_with_its_cpu_counter(self) -> None:
        """Without the `psutil` extra `-n logical` falls back to another count."""
        requirements = shlex.split(str(_step(INSTALL_STEP)["run"]))
        assert requirements[:2] == ["pip", "install"], requirements
        assert "pytest-xdist[psutil]" in requirements
        assert ".[dev]" in requirements

    def test_the_plugin_is_installed_before_the_tests_run(self) -> None:
        names = [step.get("name") for step in _jobs()["test"]["steps"]]
        assert names.index(INSTALL_STEP) < names.index(RUN_STEP)

    @pytest.mark.parametrize(
        "argument",
        [
            "--junitxml=report.xml",  # the nightly test-count badge parses it
            "--cov=soup_cli",
            "--cov-report=xml:coverage.xml",  # the Codecov upload sends it
            "--cov-report=term-missing:skip-covered",
            "-v",
            "--tb=short",
        ],
    )
    def test_the_reports_the_later_steps_read_are_still_written(self, argument: str) -> None:
        assert argument in _pytest_args()

    def test_no_step_of_any_job_spreads_one_file_over_workers(self) -> None:
        text = CI.read_text(encoding="utf-8")
        runs = [
            str(step["run"])
            for job in _jobs().values()
            for step in job.get("steps", [])
            if "run" in step
        ]
        assert runs, text[:200]
        for run in runs:
            for mode in re.findall(r"--dist[ =](\S+)", run):
                assert mode == "loadfile", run


class TestALocalRunStaysSerial:
    def _addopts(self) -> list[str]:
        text = PYPROJECT.read_text(encoding="utf-8")
        table = re.search(
            r"^\[tool\.pytest\.ini_options\]\n(.*?)(?=^\[|\Z)", text, re.M | re.S
        )
        assert table is not None
        line = re.search(r'^addopts\s*=\s*"(.*)"\s*$', table.group(1), re.M)
        assert line is not None, table.group(1)
        return shlex.split(line.group(1))

    def _extra_requirements(self) -> list[str]:
        text = PYPROJECT.read_text(encoding="utf-8")
        table = re.search(
            r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[|\Z)", text, re.M | re.S
        )
        assert table is not None
        requirements = re.findall(r'"([^"\n]*)"', table.group(1))
        assert any(requirement.startswith("pytest") for requirement in requirements)
        return requirements

    def test_the_default_options_start_no_workers(self) -> None:
        """Workers are a CI choice. In `addopts` they would also be forced on every local
        run and on every subset run."""
        args = self._addopts()
        assert _option(args, "-n") == []
        assert _option(args, "--numprocesses") == []
        assert _option(args, "--dist") == []

    def test_the_plugin_is_not_a_dev_dependency(self) -> None:
        for requirement in self._extra_requirements():
            assert "xdist" not in requirement.lower(), requirement


def test_contributing_says_what_a_parallel_run_asks_of_a_test() -> None:
    text = " ".join(CONTRIBUTING.read_text(encoding="utf-8").split())
    assert "CI runs the tests in parallel" in text
    assert "`-n logical --dist loadfile`" in text
    assert 'pip install "pytest-xdist[psutil]"' in text
