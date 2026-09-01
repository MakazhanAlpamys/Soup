"""#382 — the windows-latest illegal-instruction guard must have ONE definition.

The guard was needed a second time on 2026-08-31, when
``test_v05311.py::test_real_trl_grpotrainer_end_to_end_step`` reached a real
``trainer.train()`` and crashed the windows 3.11 cell with
``Windows fatal exception: code 0xc000001d`` — the same signature, in a file the
original guard did not cover. Copying the predicate would have been the third
time this repo shipped a duplicated source of truth (#372, #392, #424), so it
moved to ``tests/_windows_ci.py`` and both call sites import it.

These tests exist so a future third copy fails loudly instead of drifting.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

from tests._windows_ci import _windows_ci, skip_on_windows_ci

_TESTS_DIR = Path(__file__).resolve().parent
_HOME = _TESTS_DIR / "_windows_ci.py"


class TestThereIsExactlyOneDefinition:
    def test_the_predicate_is_defined_only_in_the_shared_module(self):
        """A second ``def _windows_ci`` anywhere under tests/ is the drift."""
        definers = []
        for path in sorted(_TESTS_DIR.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "_windows_ci":
                    definers.append(path.name)
        assert definers == [_HOME.name], (
            f"_windows_ci must be defined only in {_HOME.name}; found in {definers}"
        )

    def test_the_marker_is_built_only_in_the_shared_module(self):
        builders = []
        for path in sorted(_TESTS_DIR.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "skip_on_windows_ci"
                    for t in node.targets
                ):
                    builders.append(path.name)
        assert builders == [_HOME.name], (
            f"skip_on_windows_ci must be built only in {_HOME.name}; found in {builders}"
        )

    def test_both_known_call_sites_import_rather_than_redeclare(self):
        for name in ("test_v07202.py", "test_v05311.py"):
            src = (_TESTS_DIR / name).read_text(encoding="utf-8")
            assert "from tests._windows_ci import" in src, f"{name} must import the guard"


class TestTheGuardStaysNarrow:
    """A skipif that quietly widened would remove real coverage in silence.

    A skipped test and a passing test are the same colour, so both edges of the
    condition are pinned: an unknown CI CPU is excluded, a platform is not.
    """

    def test_it_fires_on_a_windows_runner(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("CI", "true")
        assert _windows_ci() is True

    def test_it_does_not_fire_on_a_local_windows_box(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delenv("CI", raising=False)
        assert _windows_ci() is False

    def test_it_does_not_fire_on_linux_ci(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("CI", "true")
        assert _windows_ci() is False

    def test_it_does_not_fire_on_macos_ci(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("CI", "true")
        assert _windows_ci() is False

    def test_the_marker_names_the_issue_and_disclaims_the_wider_reading(self):
        reason = skip_on_windows_ci.kwargs["reason"]
        assert "#382" in reason
        assert "ubuntu" in reason and "macos" in reason


class TestTheCrashingTestIsActuallyGuarded:
    """The 2026-08-31 crash site must carry the marker, not merely import it."""

    def test_the_real_grpotrainer_step_carries_the_marker(self):
        tree = ast.parse((_TESTS_DIR / "test_v05311.py").read_text(encoding="utf-8"))
        target = "test_real_trl_grpotrainer_end_to_end_step"
        found = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == target
        ]
        assert found, f"{target} not found — did it move?"
        names = {
            d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")
            for d in found[0].decorator_list
        }
        assert "skip_on_windows_ci" in names, (
            f"{target} calls a real trainer.train() and must carry the #382 guard"
        )


@pytest.mark.skipif(
    os.environ.get("CI") == "true" and sys.platform == "win32",
    reason="the control below asserts the unskipped state",
)
def test_the_guard_does_not_skip_anywhere_else():
    """Everywhere but a Windows CI runner, the marker must be inert."""
    assert _windows_ci() is False
