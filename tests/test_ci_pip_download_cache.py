"""Only the Ubuntu/Python 3.12 test cell reuses pip downloads.

The cache inputs include the project metadata and every committed constraints file,
including the compatibility floor. Checkout must precede setup-python so those inputs
can be hashed. A cache hit must never bypass installation: nightly jobs still resolve
the newest allowed versions, and the floor job still installs its constrained stack.
The remaining test cells and the four smoke/contract jobs leave pip caching disabled
to limit large dependency archives competing with the existing HF model caches.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.test_ci_full_matrix_on_approval import (
    _cell_dicts,
    _matrix,
    _on_event,
    _pull_request,
    _push,
    _render,
)

ROOT = Path(__file__).absolute().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
PROJECT_JOBS = ("test", "mlx-smoke", "torchao-contract", "pytorch-smoke", "transformers-floor")
UNCACHED_JOBS = ("mlx-smoke", "torchao-contract", "pytorch-smoke", "transformers-floor")
RUN_KINDS = (
    pytest.param((_push(), _pull_request("opened")), "QUICK", id="quick-main-and-pr"),
    pytest.param(
        (_pull_request("synchronize", labels=("ci:full",)),), "APPROVAL", id="approval"
    ),
    pytest.param((_on_event("schedule"),), "FULL", id="nightly"),
    pytest.param((_on_event("workflow_dispatch"),), "FULL", id="manual"),
    pytest.param((_push("refs/heads/release/v0.76.0"),), "RELEASE", id="release"),
)
# Literal outcomes over the entire supported 3x3 matrix, independent of the policy
# expression and of which cells a particular workflow kind currently selects.
SUPPORT_CELLS = (
    ("ubuntu-latest", "Linux", "3.10", ""),
    ("ubuntu-latest", "Linux", "3.11", ""),
    ("ubuntu-latest", "Linux", "3.12", "pip"),
    ("windows-latest", "Windows", "3.10", ""),
    ("windows-latest", "Windows", "3.11", ""),
    ("windows-latest", "Windows", "3.12", ""),
    ("macos-latest", "macOS", "3.10", ""),
    ("macos-latest", "macOS", "3.11", ""),
    ("macos-latest", "macOS", "3.12", ""),
)
RUNNER_OS = {"ubuntu-latest": "Linux", "windows-latest": "Windows", "macos-latest": "macOS"}


def _steps(job_name: str) -> list[dict[str, Any]]:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"][job_name]["steps"]


def _python_setup(job_name: str) -> tuple[int, dict[str, Any]]:
    setups = [
        (index, step)
        for index, step in enumerate(_steps(job_name))
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    ]
    assert len(setups) == 1, f"{job_name}: expected one Python setup, got {setups}"
    return setups[0]


def _project_installs(job_name: str) -> list[tuple[int, dict[str, Any]]]:
    installs = []
    for index, step in enumerate(_steps(job_name)):
        command = str(step.get("run", ""))
        if re.search(r"\bpip\s+install\b", command):
            arguments = shlex.split(command)
            if any(argument == "." or argument.startswith(".[") for argument in arguments):
                installs.append((index, step))
    assert installs, f"{job_name}: missing project installation"
    return installs


class TestPipDownloadCache:
    @pytest.mark.parametrize(("contexts", "kind"), RUN_KINDS)
    @pytest.mark.parametrize(("runner", "runner_os", "python_version", "expected"), SUPPORT_CELLS)
    def test_only_ubuntu_python312_caches_pip_for_every_run_kind(
        self,
        contexts: tuple[dict[str, Any], ...],
        kind: str,
        runner: str,
        runner_os: str,
        python_version: str,
        expected: str,
    ) -> None:
        _, setup = _python_setup("test")
        policy = setup.get("with", {}).get("cache", "")
        for context in contexts:
            cell_context = {
                **context,
                "runner": {"os": runner_os},
                "matrix": {"os": runner, "python-version": python_version},
            }
            assert _render(policy, cell_context) == expected, (
                f"{kind}: {runner}/{python_version} must select pip cache {expected!r}"
            )

    @pytest.mark.parametrize(("contexts", "kind"), RUN_KINDS)
    def test_each_selected_workflow_matrix_has_only_one_cached_cell(
        self, contexts: tuple[dict[str, Any], ...], kind: str
    ) -> None:
        _, setup = _python_setup("test")
        policy = setup.get("with", {}).get("cache", "")
        for context in contexts:
            cached = set()
            for cell in _cell_dicts(_matrix(f"{kind}_MATRIX")):
                cell_context = {**context, "runner": {"os": RUNNER_OS[cell["os"]]}, "matrix": cell}
                if _render(policy, cell_context):
                    cached.add((cell["os"], cell["python-version"]))
            assert cached == {("ubuntu-latest", "3.12")}, (kind, cached)

    @pytest.mark.parametrize("job_name", UNCACHED_JOBS)
    def test_smoke_and_contract_jobs_leave_pip_cache_disabled(self, job_name: str) -> None:
        _, setup = _python_setup(job_name)
        assert not setup.get("with", {}).get("cache"), (
            f"{job_name}: pip caching is reserved for the Ubuntu/Python 3.12 test cell"
        )
        assert "cache-dependency-path" not in setup.get("with", {}), (
            f"{job_name}: uncached setup must not declare pip cache dependency inputs"
        )

    def test_cache_inputs_cover_project_and_all_constraints(self) -> None:
        _, setup = _python_setup("test")
        dependency_paths = setup.get("with", {}).get("cache-dependency-path", "")
        assert isinstance(dependency_paths, str), "test: cache inputs must be paths"
        patterns = [line.strip() for line in dependency_paths.splitlines() if line.strip()]
        matched = {path for pattern in patterns for path in ROOT.glob(pattern) if path.is_file()}
        constraints = set((ROOT / ".github" / "constraints").glob("*.txt"))
        assert constraints, "the constraint check must cover committed constraints files"
        required = {ROOT / "pyproject.toml", *constraints}
        missing = sorted(str(path.relative_to(ROOT)) for path in required - matched)
        assert not missing, f"test: pip cache does not hash dependency inputs: {missing}"

    @pytest.mark.parametrize("job_name", PROJECT_JOBS)
    def test_checkout_precedes_python_cache_restore(self, job_name: str) -> None:
        setup_index, _ = _python_setup(job_name)
        checkouts = [
            index
            for index, step in enumerate(_steps(job_name))
            if str(step.get("uses", "")).startswith("actions/checkout@")
        ]
        assert checkouts and min(checkouts) < setup_index, (
            f"{job_name}: checkout must make cache dependency inputs available before setup-python"
        )

    @pytest.mark.parametrize("job_name", PROJECT_JOBS)
    def test_project_install_follows_cache_restore(self, job_name: str) -> None:
        setup_index, _ = _python_setup(job_name)
        for install_index, step in _project_installs(job_name):
            assert setup_index < install_index, (
                f"{job_name}: {step.get('name', 'project install')} must follow setup-python"
            )

    @pytest.mark.parametrize("job_name", PROJECT_JOBS)
    def test_cache_hit_cannot_bypass_project_install(self, job_name: str) -> None:
        for _, step in _project_installs(job_name):
            assert "if" not in step, f"{job_name}: project installation must be unconditional"
            assert "cache-hit" not in str(step["run"]), (
                f"{job_name}: cache hits must not gate installation inside the shell"
            )
