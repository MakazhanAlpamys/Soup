"""Every light-core dependency is installed at its declared floor in CI (#1802).

The training floor job cannot install old core dependencies that the training stack
requires newer versions of. The separate core-floor job installs no extra, checks
the resolved versions, and builds the entire command tree and starts light commands.
These guards run on newest-dependency jobs too, so changing a declaration, pin, or CI
command alone fails even before the real floor cell runs.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from tests.test_click_argument_help import _core_requirements

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ".github/constraints/core-floor.txt"
REQUIRED_TESTS = {
    "tests/test_core_dependency_floors.py",
    "tests/test_cli_dependency_floors.py",
    "tests/test_click_argument_help.py",
    "tests/test_cli_startup_is_light.py",
    "tests/test_cli.py",
    "tests/test_cli_subprocess.py",
    "tests/test_plotext_compat.py",
}
# PyPI has no stable release at these declarations. Values are the first
# non-yanked stable releases pip resolves (measured 2026-10-15; Hub 0.16.1 is yanked).
FIRST_PUBLISHED = {
    ("huggingface-hub", Version("0.16.0")): "0.16.2",
    ("plotext", Version("5.2.0")): "5.2.2",
}


def _pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in (ROOT / CONSTRAINTS).read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        requirement = Requirement(raw)
        name = canonicalize_name(requirement.name)
        specs = list(requirement.specifier)
        assert len(specs) == 1 and specs[0].operator == "==", raw
        assert not requirement.extras and requirement.marker is None and requirement.url is None
        assert "*" not in specs[0].version, raw
        assert name not in pins, f"duplicate core-floor pin: {name}"
        pins[name] = specs[0].version
    return pins


def _job() -> dict:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert "core-floor" in workflow["jobs"], "CI must run the declared light-core floors"
    return workflow["jobs"]["core-floor"]


def _version_check() -> str:
    steps = _job()["steps"]
    checks = [step for step in steps if step.get("name") == "Assert core floor versions"]
    assert len(checks) == 1, "core-floor must check what pip actually installed"
    assert checks[0].get("shell") == "python"
    return checks[0]["run"]


def test_every_core_dependency_is_pinned_to_its_declared_floor() -> None:
    declared = {canonicalize_name(req.name): req for req in _core_requirements()}
    pins = _pins()
    assert pins.keys() == declared.keys(), "pin every core dependency, and no training extra"
    for name, requirement in declared.items():
        floors = [spec.version for spec in requirement.specifier if spec.operator == ">="]
        assert len(floors) == 1, f"{name} needs one inclusive declared floor"
        expected = FIRST_PUBLISHED.get((name, Version(floors[0])), floors[0])
        assert Version(pins[name]) == Version(expected), (
            f"{name}: core-floor installs {pins[name]}, declared floor is {floors[0]}"
        )
        assert Version(pins[name]) in requirement.specifier


def test_floor_job_installs_only_core_under_its_constraints() -> None:
    runs = [str(step.get("run", "")) for step in _job()["steps"]]
    installs = [run for run in runs if "pip install" in run]
    assert len(installs) == 1, "a later pip install can silently upgrade a core floor"
    words = shlex.split(installs[0])
    assert words[words.index("-e") + 1] == ".", "[dev] pulls in incompatible training deps"
    assert words[words.index("-c") + 1] == CONSTRAINTS
    assert not any("[" in word for word in words), "install no extras in the core floor cell"
    assert "python -m pip check" in runs


def test_floor_job_isolates_pins_from_runner_tooling() -> None:
    steps = _job()["steps"]
    install_index = next(index for index, step in enumerate(steps)
                         if "pip install" in step.get("run", ""))
    setup = "\n".join(step.get("run", "") for step in steps[:install_index])
    # A runner's wheel>=0.46 requires packaging>=24; Soup must exercise 21.0.
    # PATH must carry the venv into every subsequent pip, Python and pytest step.
    assert 'python -m venv "$RUNNER_TEMP/soup-core-floor"' in setup
    assert '"$RUNNER_TEMP/soup-core-floor/bin" >> "$GITHUB_PATH"' in setup
    assert "--system-site-packages" not in setup


def test_floor_job_runs_command_tree_help_and_startup_checks() -> None:
    arguments: list[str] = []
    for step in _job()["steps"]:
        run = str(step.get("run", "")).strip()
        if run.startswith(("pytest ", "python -m pytest ")):
            words = shlex.split(run)
            arguments.extend(words[words.index("pytest") + 1 :])
    assert REQUIRED_TESTS <= set(arguments), f"core-floor must run {sorted(REQUIRED_TESTS)}"
    assert not {"-k", "-m", "--deselect"} & set(arguments), "do not hide floor failures"


def test_installed_version_check_rejects_every_upgraded_pin() -> None:
    script = _version_check()
    pins = _pins()
    constraints = (ROOT / CONSTRAINTS).read_text(encoding="utf-8")
    with (
        patch("pathlib.Path.read_text", return_value=constraints),
        patch("importlib.util.find_spec", return_value=None),
        patch("importlib.metadata.version", side_effect=pins.__getitem__),
    ):
        exec(compile(script, "<core-floor version check>", "exec"), {})
        for name, version in pins.items():
            pins[name] = "999.0"
            with pytest.raises(AssertionError, match=re.escape(name)):
                exec(compile(script, "<core-floor version check>", "exec"), {})
            pins[name] = version


def test_core_floor_check_rejects_an_installed_training_stack() -> None:
    script = _version_check()
    with (
        patch("pathlib.Path.read_text", return_value=(ROOT / CONSTRAINTS).read_text("utf-8")),
        patch("importlib.metadata.version", side_effect=_pins().__getitem__),
        patch(
            "importlib.util.find_spec",
            side_effect=lambda name: object() if name == "torch" else None,
        ),
    ):
        with pytest.raises(AssertionError, match="torch"):
            exec(compile(script, "<core-floor version check>", "exec"), {})
