"""Linux CI jobs install the CPU-only torch wheel; Windows and macOS cells are left alone.

PyPI's Linux ``torch`` wheel depends on the CUDA runtime (19 packages, 2.43 GB beside the
555 MB wheel, for torch 2.14.1), and no CI runner has a GPU. PyTorch's own index serves a
196 MB ``2.14.1+cpu`` wheel with none of it. Windows and macOS get CPU wheels from PyPI
already, so only Linux needs the second index.

The switch is ``PIP_EXTRA_INDEX_URL`` in the ``env:`` of the install step, never a flag on
the command: the ``pip install`` lines are parsed and pinned by
``tests/test_ci_full_matrix_on_approval.py``, and an environment variable leaves them as
they are. PyPI stays the main index (``PIP_INDEX_URL`` is never set), because the CPU
index mirrors only older or equal copies of every other package. This file pins:

* every step that installs torch (the name, or an extra that pulls it in) sets the CPU
  index on a Linux runner and sets nothing on Windows and macOS, matrix cells included;
* the jobs that stay on PyPI's wheel are named, with the reason, and the set of jobs that
  install torch is exact, so a new one has to decide;
* the index is set on those steps only: not for a whole job, where it would also redirect
  a pip that a test runs, and not for a workflow;
* the floor job's version assertion reads the public version, because the CPU wheel
  reports ``2.6.0+cpu`` for the pin ``torch==2.6.0``.

Read off the parsed workflows, never off the text. The control class mutates copies of the
real workflows and expects each mutation to be reported.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from tests.test_ci_full_matrix_on_approval import (
    _PIP_INSTALL,
    _render,
    _requested,
    _shell_commands,
    _workflows,
)
from tests.test_ci_pinned_upstream import _declared

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CONSTRAINTS = ROOT / ".github" / "constraints"

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
INDEX_VARIABLES = ("PIP_EXTRA_INDEX_URL", "PIP_INDEX_URL")
#: A pip install that asks for either of these brings torch with it.
TORCH_PACKAGES = frozenset({"torch", "torchaudio"})
RUNNER_OS = {
    "ubuntu-latest": "Linux",
    "windows-latest": "Windows",
    "macos-latest": "macOS",
    "macos-14": "macOS",
}

#: (workflow file, job) pairs whose install step sets the CPU index on Linux.
ON_CPU_INDEX = frozenset(
    {
        ("ci.yml", "test"),
        ("ci.yml", "pytorch-smoke"),
        ("ci.yml", "transformers-floor"),
        ("recipe-validation.yml", "validate-recipes"),
        ("recipe-attach.yml", "verify"),
    }
)
#: Jobs that install torch and stay on PyPI's wheel, each with why.
LEFT_ON_PYPI = {
    ("ci.yml", "torchao-contract"): (
        "the CPU index also hosts torchao, as 0.18.0+cpu (pure Python), and pip prefers a "
        "local version to PyPI's plain 0.18.0 (the compiled abi3 wheel): the job would test "
        "a build no user installs from PyPI"
    ),
    ("dependency-drift.yml", "latest-stack"): (
        "the weekly job that resolves what a Linux user gets from PyPI"
    ),
}


def _cells(workflows: dict[str, Any], file: str, job: dict[str, Any]) -> list[str]:
    """The runner labels *job* runs on: the matrix's operating systems for ``test``."""
    runs_on = job["runs-on"]
    if runs_on == "${{ matrix.os }}":
        release = workflows[file]["jobs"]["plan"]["env"]["RELEASE_MATRIX"]
        return list(json.loads(release)["os"])
    return [runs_on]


def _context(label: str) -> dict[str, Any]:
    return {
        "github": {"event_name": "pull_request"},
        "needs": {
            "plan": {"outputs": {"constraints": "-c .github/constraints/pinned-upstream.txt"}}
        },
        "runner": {"os": RUNNER_OS[label]},
        "matrix": {"os": label},
    }


def _installs_torch(rendered: str, pyproject: str) -> bool:
    """Whether a rendered ``pip install`` names torch or installs an extra that needs it."""
    names, extras = _requested(rendered)
    if names & TORCH_PACKAGES:
        return True
    return extras is not None and bool(set(_declared(pyproject, extras)) & TORCH_PACKAGES)


def _torch_steps(workflows: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """(file, job name, step) for every step whose pip install brings torch."""
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    found = []
    for file, workflow in workflows.items():
        for name, job in workflow["jobs"].items():
            for step in job.get("steps", []):
                if "python" in str(step.get("shell", "")):
                    continue  # a script for another interpreter, not a command line
                for command in _shell_commands(str(step.get("run", ""))):
                    if not _PIP_INSTALL.search(command):
                        continue
                    if _installs_torch(_render(command, _context("ubuntu-latest")), pyproject):
                        found.append((file, name, step))
                        break
    return found


def _index_of(step: dict[str, Any], label: str) -> str:
    """The extra index *step* gives pip on a runner labelled *label*; empty if none."""
    raw = (step.get("env") or {}).get("PIP_EXTRA_INDEX_URL")
    return "" if raw is None else _render(str(raw), _context(label))


def _problems(workflows: dict[str, Any]) -> list[str]:
    """Everything wrong with how *workflows* pick torch's index; empty if they are right."""
    problems = []
    torch_steps = _torch_steps(workflows)
    for file, name, step in torch_steps:
        job = workflows[file]["jobs"][name]
        for label in _cells(workflows, file, job):
            linux = RUNNER_OS[label] == "Linux"
            want = CPU_INDEX if linux and (file, name) not in LEFT_ON_PYPI else ""
            got = _index_of(step, label)
            if got != want:
                problems.append(
                    f"{file} / {name} / {step.get('name')!r} on {label}: "
                    f"PIP_EXTRA_INDEX_URL is {got!r}, expected {want!r}"
                )
    in_scope = {id(step) for _, _, step in torch_steps}
    for file, workflow in workflows.items():
        for variable in INDEX_VARIABLES:
            if variable in (workflow.get("env") or {}):
                problems.append(f"{file}: {variable} is set for the whole workflow")
        for name, job in workflow["jobs"].items():
            for variable in INDEX_VARIABLES:
                if variable in (job.get("env") or {}):
                    problems.append(f"{file} / {name}: {variable} is set for the whole job")
            for step in job.get("steps", []):
                environment = step.get("env") or {}
                if "PIP_INDEX_URL" in environment:
                    problems.append(
                        f"{file} / {name} / {step.get('name')!r}: PIP_INDEX_URL drops PyPI as "
                        "the main index; use PIP_EXTRA_INDEX_URL"
                    )
                if "PIP_EXTRA_INDEX_URL" in environment and id(step) not in in_scope:
                    problems.append(
                        f"{file} / {name} / {step.get('name')!r}: sets PIP_EXTRA_INDEX_URL but "
                        "installs no torch"
                    )
                if "download.pytorch.org" in str(step.get("run", "")):
                    problems.append(
                        f"{file} / {name} / {step.get('name')!r}: names PyTorch's index on a "
                        "command line; set PIP_EXTRA_INDEX_URL in the step's env instead"
                    )
    return problems


def _install_step(workflows: dict[str, Any], file: str, job: str) -> dict[str, Any]:
    (step,) = [s for f, j, s in _torch_steps(workflows) if (f, j) == (file, job)]
    return step


class TestTheReaderHasTeeth:
    """CONTROL for the tests below: each mutation of a copy of the real workflows is reported."""

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ('pip install -e ".[dev]"', True),
            ('pip install -e ".[train]" -c pins.txt', True),
            ('pip install -e ".[dev,audio]" -c floor.txt', True),
            ('pip install -e ".[dev,qat]" "torch>=2.6.0"', True),
            ('pip install -e "." "torch>=2.6.0"', True),
            ("pip install -e .", False),
            ("pip install -e . pytest", False),
            ('pip install -e ".[mlx]" pytest', False),
            ("pip install ruff", False),
            ("pip install -e . huggingface-hub pyyaml", False),
        ],
    )
    def test_it_knows_what_installs_torch(self, command, expected):
        pyproject = PYPROJECT.read_text(encoding="utf-8")
        assert _installs_torch(command, pyproject) is expected, command

    def test_the_test_job_runs_on_all_three_systems_and_only_linux_gets_the_index(self):
        workflows = _workflows()
        job = workflows["ci.yml"]["jobs"]["test"]
        assert sorted(_cells(workflows, "ci.yml", job)) == [
            "macos-latest",
            "ubuntu-latest",
            "windows-latest",
        ]
        step = _install_step(workflows, "ci.yml", "test")
        assert _index_of(step, "ubuntu-latest") == CPU_INDEX
        assert _index_of(step, "windows-latest") == ""
        assert _index_of(step, "macos-latest") == ""

    def test_a_missing_index_on_a_linux_job_is_reported(self):
        workflows = copy.deepcopy(_workflows())
        _install_step(workflows, "ci.yml", "pytorch-smoke")["env"].pop("PIP_EXTRA_INDEX_URL")
        (problem,) = _problems(workflows)
        assert "ci.yml / pytorch-smoke" in problem and "expected" in problem, problem

    def test_a_missing_index_on_the_linux_test_cell_is_reported(self):
        workflows = copy.deepcopy(_workflows())
        _install_step(workflows, "ci.yml", "test")["env"].pop("PIP_EXTRA_INDEX_URL")
        (problem,) = _problems(workflows)
        assert "ci.yml / test" in problem and "ubuntu-latest" in problem, problem

    def test_the_index_on_every_system_is_reported_for_windows_and_macos(self):
        workflows = copy.deepcopy(_workflows())
        _install_step(workflows, "ci.yml", "test")["env"]["PIP_EXTRA_INDEX_URL"] = CPU_INDEX
        problems = _problems(workflows)
        assert len(problems) == 2, problems
        assert any("windows-latest" in problem for problem in problems), problems
        assert any("macos-latest" in problem for problem in problems), problems

    def test_a_wrong_index_is_reported(self):
        workflows = copy.deepcopy(_workflows())
        step = _install_step(workflows, "recipe-attach.yml", "verify")
        step["env"]["PIP_EXTRA_INDEX_URL"] = "https://download.pytorch.org/whl/cu130"
        (problem,) = _problems(workflows)
        assert "recipe-attach.yml / verify" in problem, problem

    def test_an_exempt_job_that_takes_the_index_is_reported(self):
        workflows = copy.deepcopy(_workflows())
        step = _install_step(workflows, "ci.yml", "torchao-contract")
        step["env"] = {"PIP_EXTRA_INDEX_URL": CPU_INDEX}
        (problem,) = _problems(workflows)
        assert "torchao-contract" in problem and "expected ''" in problem, problem

    def test_a_new_job_that_installs_torch_is_reported_until_it_has_the_index(self):
        workflows = copy.deepcopy(_workflows())
        workflows["ci.yml"]["jobs"]["new-job"] = {
            "runs-on": "ubuntu-latest",
            "steps": [{"name": "Install", "run": 'pip install -e ".[train]"'}],
        }
        (problem,) = _problems(workflows)
        assert "new-job" in problem, problem
        workflows["ci.yml"]["jobs"]["new-job"]["steps"][0]["env"] = {
            "PIP_EXTRA_INDEX_URL": CPU_INDEX
        }
        assert _problems(workflows) == []

    def test_the_index_for_a_whole_job_or_workflow_is_reported(self):
        for place in ("job", "workflow"):
            workflows = copy.deepcopy(_workflows())
            target = workflows["ci.yml"]["jobs"]["lint"] if place == "job" else workflows["ci.yml"]
            target.setdefault("env", {})["PIP_EXTRA_INDEX_URL"] = CPU_INDEX
            (problem,) = _problems(workflows)
            assert f"whole {place}" in problem, problem

    def test_pypi_as_the_main_index_dropped_is_reported(self):
        workflows = copy.deepcopy(_workflows())
        step = _install_step(workflows, "ci.yml", "pytorch-smoke")
        step["env"]["PIP_INDEX_URL"] = CPU_INDEX
        (problem,) = _problems(workflows)
        assert "PIP_INDEX_URL drops PyPI" in problem, problem

    def test_the_index_on_a_step_that_installs_no_torch_is_reported(self):
        workflows = copy.deepcopy(_workflows())
        (step,) = [
            s
            for s in workflows["ci.yml"]["jobs"]["lint"]["steps"]
            if str(s.get("run", "")) == "pip install ruff"
        ]
        step["env"] = {"PIP_EXTRA_INDEX_URL": CPU_INDEX}
        (problem,) = _problems(workflows)
        assert "installs no torch" in problem, problem

    def test_the_index_spelled_on_a_command_line_is_reported(self):
        workflows = copy.deepcopy(_workflows())
        step = _install_step(workflows, "ci.yml", "pytorch-smoke")
        step["run"] += f" --extra-index-url {CPU_INDEX}"
        problems = _problems(workflows)
        assert any("names PyTorch's index on a command line" in line for line in problems), problems


class TestTheCommittedWorkflows:
    def test_every_step_that_installs_torch_picks_its_index_by_operating_system(self):
        assert _problems(_workflows()) == []

    def test_the_jobs_that_install_torch_are_exactly_the_ones_named_here(self):
        """A new job that installs torch is on neither list until someone decides."""
        installing = {(file, name) for file, name, _ in _torch_steps(_workflows())}
        assert installing == ON_CPU_INDEX | LEFT_ON_PYPI.keys()
        assert not ON_CPU_INDEX & LEFT_ON_PYPI.keys()

    def test_each_job_on_the_cpu_index_installs_on_linux(self):
        workflows = _workflows()
        for file, name in sorted(ON_CPU_INDEX):
            job = workflows[file]["jobs"][name]
            assert "Linux" in {RUNNER_OS[label] for label in _cells(workflows, file, job)}, name

    @pytest.mark.parametrize("file", ["pinned-upstream.txt", "transformers-floor.txt"])
    def test_the_torch_pins_are_public_versions_a_cpu_wheel_satisfies(self, file):
        """``torch==2.6.0+cu124`` would refuse the CPU wheel; ``torch==2.6.0`` accepts it."""
        text = (CONSTRAINTS / file).read_text(encoding="utf-8")
        pins = dict(re.findall(r"^([A-Za-z0-9_-]+)==(\S+)\s*$", text, re.M))
        assert "torch" in pins, file
        for name in sorted(TORCH_PACKAGES & pins.keys()):
            assert Version(pins[name]).local is None, (file, name, pins[name])
            assert SpecifierSet(f"=={pins[name]}").contains(f"{pins[name]}+cpu"), (file, name)


def _floor_script() -> str:
    """The Python the floor job runs to compare installed versions with its constraints."""
    steps = _workflows()["ci.yml"]["jobs"]["transformers-floor"]["steps"]
    (step,) = [
        s for s in steps if s.get("name") == "Assert installed versions match floor constraints"
    ]
    match = re.search(r"<<'PY'\n(.*?)\nPY\s*$", str(step["run"]), re.S)
    assert match is not None, step["run"]
    return match.group(1)


def _run_floor_script(monkeypatch: pytest.MonkeyPatch, versions: dict[str, str]) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: versions[name])
    exec(compile(_floor_script(), "<transformers-floor assertion>", "exec"), {"__name__": "x"})


class TestTheFloorJobReadsThePublicVersion:
    """``torch==2.6.0`` installs ``2.6.0+cpu`` from the CPU index, and ``md.version`` says so."""

    @staticmethod
    def _pins() -> dict[str, str]:
        text = (CONSTRAINTS / "transformers-floor.txt").read_text(encoding="utf-8")
        return dict(re.findall(r"^([A-Za-z0-9_-]+)==([^\s#]+)\s*$", text, re.M))

    def test_the_exact_pins_pass(self, monkeypatch):
        _run_floor_script(monkeypatch, self._pins())

    def test_the_cpu_wheels_pass(self, monkeypatch):
        pins = self._pins()
        versions = {
            name: f"{pin}+cpu" if name in TORCH_PACKAGES else pin for name, pin in pins.items()
        }
        assert versions["torch"].endswith("+cpu") and versions["torchaudio"].endswith("+cpu")
        _run_floor_script(monkeypatch, versions)

    @pytest.mark.parametrize(
        ("name", "drifted"),
        [
            ("torch", "2.6.1+cpu"),
            ("torch", "2.6.1"),
            ("torchaudio", "2.7.0+cpu"),
            ("transformers", "5.16.2"),
            ("typer", "0.16.0"),
        ],
    )
    def test_a_drifted_version_still_fails(self, monkeypatch, name, drifted):
        versions = dict(self._pins())
        assert versions[name] != drifted
        versions[name] = drifted
        with pytest.raises(AssertionError, match=f"floor job expected {name}=="):
            _run_floor_script(monkeypatch, versions)
