"""No test cell of CI caches pip downloads.

The ubuntu / Python 3.12 cell used to restore a pip cache (#1970). Measured on 2026-10-19
over the last 12 pull-request runs, that cost more than it saved: restoring it took 0.49 min
(0.33 to 0.60) and the install then 1.54 min, 2.03 min in all, where the uncached ubuntu /
3.10 cell installed in 1.80 min. The cache entries were also 3 GB each (two of them, a new
one whenever ``pyproject.toml`` or a constraints file changes) and filled 6.1 GB of the
8.8 GB the repository used of its 10 GB Actions cache, which is what evicted the small
Hugging Face caches next to them. So no cell declares ``cache:`` or ``cache-dependency-path:``
any more, and these tests pin that for every kind of run and every cell of the support matrix.

What stays pinned: a cache must never bypass installation (nightly jobs still resolve the
newest allowed versions, the floor job still installs its constrained stack), and the
project install follows ``setup-python``. A job that wants a pip cache again has to bring a
measurement that beats the one above, and change this file with it.
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
#: The only `if:` a project install may have: none, or the docs-only guard of its job.
DOCS_ONLY_GUARDS = ("", "env.RUN_SUITE == 'true'", "needs.plan.outputs.docs_only != 'true'")
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
# expression and of which cells a particular workflow kind currently selects: none caches.
SUPPORT_CELLS = (
    ("ubuntu-latest", "Linux", "3.11"),
    ("ubuntu-latest", "Linux", "3.12"),
    ("ubuntu-latest", "Linux", "3.13"),
    ("windows-latest", "Windows", "3.11"),
    ("windows-latest", "Windows", "3.12"),
    ("windows-latest", "Windows", "3.13"),
    ("macos-latest", "macOS", "3.11"),
    ("macos-latest", "macOS", "3.12"),
    ("macos-latest", "macOS", "3.13"),
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


class TestNoPipDownloadCache:
    @pytest.mark.parametrize(("contexts", "kind"), RUN_KINDS)
    @pytest.mark.parametrize(("runner", "runner_os", "python_version"), SUPPORT_CELLS)
    def test_no_cell_caches_pip_for_any_run_kind(
        self,
        contexts: tuple[dict[str, Any], ...],
        kind: str,
        runner: str,
        runner_os: str,
        python_version: str,
    ) -> None:
        _, setup = _python_setup("test")
        policy = setup.get("with", {}).get("cache", "")
        for context in contexts:
            cell_context = {
                **context,
                "runner": {"os": runner_os},
                "matrix": {"os": runner, "python-version": python_version},
            }
            assert _render(policy, cell_context) == "", (
                f"{kind}: {runner}/{python_version} must not select a pip cache"
            )

    @pytest.mark.parametrize(("contexts", "kind"), RUN_KINDS)
    def test_each_selected_workflow_matrix_has_no_cached_cell(
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
            assert cached == set(), (kind, cached)

    @pytest.mark.parametrize("job_name", PROJECT_JOBS)
    def test_no_project_job_declares_a_pip_cache_or_its_inputs(self, job_name: str) -> None:
        _, setup = _python_setup(job_name)
        assert "cache" not in setup.get("with", {}), (
            f"{job_name}: pip caching is off (2026-10-19: the cache cost more than it saved)"
        )
        assert "cache-dependency-path" not in setup.get("with", {}), (
            f"{job_name}: no pip cache, so no cache dependency inputs"
        )

    @pytest.mark.parametrize("job_name", PROJECT_JOBS)
    def test_checkout_precedes_python_setup(self, job_name: str) -> None:
        setup_index, _ = _python_setup(job_name)
        checkouts = [
            index
            for index, step in enumerate(_steps(job_name))
            if str(step.get("uses", "")).startswith("actions/checkout@")
        ]
        assert checkouts and min(checkouts) < setup_index, (
            f"{job_name}: checkout must come before setup-python (a cache hashes the project files)"
        )

    @pytest.mark.parametrize("job_name", PROJECT_JOBS)
    def test_project_install_follows_python_setup(self, job_name: str) -> None:
        setup_index, _ = _python_setup(job_name)
        for install_index, step in _project_installs(job_name):
            assert setup_index < install_index, (
                f"{job_name}: {step.get('name', 'project install')} must follow setup-python"
            )

    @pytest.mark.parametrize("job_name", PROJECT_JOBS)
    def test_cache_hit_cannot_bypass_project_install(self, job_name: str) -> None:
        for _, step in _project_installs(job_name):
            # 2026-10-20: the one condition a project install may carry is the docs-only
            # guard (tests/test_ci_docs_only_fast_path.py): it skips every step of a cell or
            # smoke job that does nothing in a docs-only pull request. Never a cache result.
            assert str(step.get("if", "")) in DOCS_ONLY_GUARDS, (
                f"{job_name}: project installation must be unconditional"
            )
            assert "cache-hit" not in str(step["run"]), (
                f"{job_name}: cache hits must not gate installation inside the shell"
            )
