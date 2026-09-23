"""#382 — the windows-latest illegal-instruction guard must have ONE definition.

The guard was needed a second time on 2026-08-31, when
``test_v05311.py::test_real_trl_grpotrainer_end_to_end_step`` reached a real
``trainer.train()`` and crashed the windows 3.11 cell with
``Windows fatal exception: code 0xc000001d`` — the same signature, in a file the
original guard did not cover. Copying the predicate would have been the third
time this repo shipped a duplicated source of truth (#372, #392, #424), so it
moved to ``tests/_windows_ci.py`` and both call sites import it.

These tests exist so a future copy fails loudly instead of drifting. Four files
carry the guard now, and each one's guarded tests are named in
:data:`GUARDED_TESTS` so a deleted decorator fails here too.
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

#: The files that gate a real ``trainer.train()`` behind the shared guard. Four
#: now: #382 (v07202), the 2026-08-31 GRPO crash (v05311), and the two from
#: #1018's rewind batch (#1059 and #1062).
KNOWN_CALL_SITES = (
    "test_v07202.py",
    "test_v05311.py",
    "test_rewind_hf.py",
    "test_rewind_wiring.py",
)


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

    def test_every_known_call_site_imports_rather_than_redeclares(self):
        for name in KNOWN_CALL_SITES:
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


#: Every test that carries the guard today, by file. A name here must exist AND
#: be decorated: the call-site test below only greps for the import line, so
#: deleting a decorator while leaving the import passed it (#1062 review). That
#: is the same silent-drift failure the guard itself exists to prevent, one
#: level up.
GUARDED_TESTS: dict[str, tuple[str, ...]] = {
    "test_v05311.py": ("test_real_trl_grpotrainer_end_to_end_step",),
    "test_v07202.py": (
        "test_one_nf4_training_step_actually_runs",
        "test_the_saved_adapter_is_canonical",
    ),
    "test_rewind_hf.py": (
        "test_recorded_rows_match_sampler_order",
        "test_grad_accum_step_and_micro_sequence",
        "test_eval_passes_record_nothing",
        "test_rows_recorded_with_dataloader_workers",
        "test_factory_is_cached_and_unattached_trainer_records_nothing",
        "test_row_losses_failure_disables_recorder_without_stopping_training",
        "test_resumed_run_disables_the_recorder",
        "test_packing_is_refused_and_unpacked_control_records",
    ),
    "test_rewind_wiring.py": ("test_hf_training_run_records_every_row_once",),
}


def _decorator_names(node: ast.AST) -> set:
    return {
        d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")
        for d in node.decorator_list
    }


class TestTheCrashingTestsAreActuallyGuarded:
    """Each crash site must CARRY the marker, not merely import it."""

    @pytest.mark.parametrize(
        "filename, target",
        [(f, t) for f, targets in GUARDED_TESTS.items() for t in targets],
    )
    def test_the_guarded_test_carries_the_marker(self, filename, target):
        tree = ast.parse((_TESTS_DIR / filename).read_text(encoding="utf-8"))
        found = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == target
        ]
        assert found, f"{filename}::{target} not found — did it move or get renamed?"
        assert "skip_on_windows_ci" in _decorator_names(found[0]), (
            f"{filename}::{target} runs a real trainer.train() and must carry the "
            f"#382 guard"
        )

    def test_every_guarded_file_is_a_known_call_site(self):
        """The two lists cannot drift apart."""
        assert set(GUARDED_TESTS) == set(KNOWN_CALL_SITES)


@pytest.mark.skipif(
    os.environ.get("CI") == "true" and sys.platform == "win32",
    reason="the control below asserts the unskipped state",
)
def test_the_guard_does_not_skip_anywhere_else():
    """Everywhere but a Windows CI runner, the marker must be inert."""
    assert _windows_ci() is False
