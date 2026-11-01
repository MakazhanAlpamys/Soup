"""#382 — the windows-latest illegal-instruction guard must have ONE definition.

The guard was needed a second time on 2026-08-31, when
``test_v05311.py::test_real_trl_grpotrainer_end_to_end_step`` reached a real
``trainer.train()`` and crashed the windows 3.11 cell with
``Windows fatal exception: code 0xc000001d`` — the same signature, in a file the
original guard did not cover. Copying the predicate would have been the third
time this repo shipped a duplicated source of truth (#372, #392, #424), so it
moved to ``tests/_windows_ci.py`` and both call sites import it.

These tests exist so a future copy fails loudly instead of drifting. One file
carries the guard now: the two NF4 tests in ``test_v07202.py``, whose crash is
the bitsandbytes wheel. The others traced to CPU bf16/fp16 matmuls (#1314) and
request ``aten_half_matmuls`` instead (#1359, #2307). The sets are named here, in
:data:`GUARDED_TESTS` and :data:`FIXTURE_TESTS`, so a deleted decorator, a dropped
fixture or a skip in a file not named here fails here too.
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

#: The files that gate a crashing native kernel behind the shared guard: #382
#: (v07202, an NF4 ``trainer.train()`` whose crash is the bitsandbytes wheel). v05311,
#: test_rewind_hf and test_rewind_wiring dropped it in #1359, once #1314 traced their
#: crash to CPU half-precision matmuls. test_expert_stream (run 40003126970),
#: test_stream_loss_chunk (run 40014485149), test_bench_train_qlora_adapters (run
#: 40761798582), test_issue1219_classifier_data_loader and test_issue1220_resume_prm_moe
#: dropped it in #2307 and request ``aten_half_matmuls`` instead.
KNOWN_CALL_SITES = ("test_v07202.py",)


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

    def test_no_other_test_file_uses_the_guard(self):
        """A skip in a file not named in KNOWN_CALL_SITES is pinned by nothing here, so it
        could widen or outlive its crash in silence (#2307 found two such files)."""
        users = sorted(
            path.name
            for path in _TESTS_DIR.rglob("*.py")
            if path.name not in (_HOME.name, Path(__file__).name)
            and "from tests._windows_ci import" in path.read_text(encoding="utf-8")
        )
        assert users == sorted(KNOWN_CALL_SITES), (
            f"the #382 guard is imported in {users}; only {list(KNOWN_CALL_SITES)} may "
            f"use it. Request aten_half_matmuls for a CPU bf16/fp16 crash, or add the file "
            f"to KNOWN_CALL_SITES and GUARDED_TESTS with the run and its crash frame"
        )


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
    "test_v07202.py": (
        "test_one_nf4_training_step_actually_runs",
        "test_the_saved_adapter_is_canonical",
    ),
}

#: Every test that runs a real CPU bf16/fp16 step and requests
#: ``aten_half_matmuls`` instead of the skip (#1314, #1359). Dropping the fixture
#: from one of them sends its matmuls back to oneDNN / MKL, which is the path that
#: crashes part of the windows-latest fleet, so each must keep requesting it.
#: ``test_rewind_wiring.py`` is not here: its run trains fp32 and never needed it.
FIXTURE_TESTS: dict[str, tuple[str, ...]] = {
    "test_expert_stream.py": (
        "test_streamed_experts_backward_preserves_forward_autocast",
        "test_two_read_experts_equal_the_resident_model_and_the_default_path",
    ),
    "test_stream_loss_chunk.py": (
        "test_they_equal_the_full_logits_step_within_the_tolerance",
        "test_one_chunk_is_the_full_step_exactly",
        "test_the_token_losses_give_the_rewind_logs_row_losses",
    ),
    "test_bench_train_qlora_adapters.py": ("test_a_run_with_bf16_adapters_is_measured_and_valid",),
    "test_issue1219_classifier_data_loader.py": (
        "test_documented_classifier_example_trains_and_records_labels",
    ),
    "test_issue1220_resume_prm_moe.py": (
        "test_prm_resume_execution_and_checkpoints",
        "test_mole_resume_execution_checkpoints_and_gate",
    ),
    "test_v05311.py": ("test_real_trl_grpotrainer_end_to_end_step",),
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
}


def _decorator_names(node: ast.AST) -> set:
    return {
        d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")
        for d in node.decorator_list
    }


def _param_names(node: ast.FunctionDef) -> set:
    return {arg.arg for arg in node.args.args}


def _is_call_to(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and getattr(node.func, "attr", "") == name


def _find_test(filename: str, target: str) -> ast.FunctionDef:
    tree = ast.parse((_TESTS_DIR / filename).read_text(encoding="utf-8"))
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == target
    ]
    assert found, f"{filename}::{target} not found — did it move or get renamed?"
    return found[0]


class TestTheCrashingTestsAreActuallyGuarded:
    """Each crash site must CARRY the marker, not merely import it."""

    @pytest.mark.parametrize(
        "filename, target",
        [(f, t) for f, targets in GUARDED_TESTS.items() for t in targets],
    )
    def test_the_guarded_test_carries_the_marker(self, filename, target):
        node = _find_test(filename, target)
        assert "skip_on_windows_ci" in _decorator_names(node), (
            f"{filename}::{target} runs a real trainer.train() and must carry the "
            f"#382 guard"
        )

    def test_every_guarded_file_is_a_known_call_site(self):
        """The lists cannot drift apart."""
        assert set(GUARDED_TESTS) == set(KNOWN_CALL_SITES)


def _marker_condition(platform: str, ci: str | None) -> bool:
    """The condition ``skip_on_windows_ci`` has on a host with this platform and ``CI``.

    The marker's condition is computed once, when ``tests/_windows_ci.py`` is
    imported, so patching the platform afterwards changes nothing. Executing the
    module's source in a throwaway namespace evaluates it again without touching
    the real module (a ``reload`` would swap the marker object that every test file
    already holds).
    """
    source = _HOME.read_text(encoding="utf-8")
    namespace = {"__name__": "_windows_ci_probe"}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "platform", platform)
        if ci is None:
            patch.delenv("CI", raising=False)
        else:
            patch.setenv("CI", ci)
        exec(compile(source, str(_HOME), "exec"), namespace)
    return namespace["skip_on_windows_ci"].args[0]


class TestTheWindowsCiSkipStaysNarrow:
    """The skip on the NF4 tests must exclude one CI fleet, nothing more.

    A skip that quietly widened would remove the tests from ubuntu and macos too and
    look exactly like a pass, so both edges of the condition are pinned on the marker
    itself, not only on ``_windows_ci``.
    """

    def test_it_skips_on_a_windows_runner(self):
        assert _marker_condition("win32", "true") is True

    def test_it_runs_on_a_local_windows_box(self):
        """The maintainer's Windows CPU is known to run these tests."""
        assert _marker_condition("win32", None) is False

    def test_it_runs_on_linux_ci(self):
        assert _marker_condition("linux", "true") is False

    def test_it_runs_on_macos_ci(self):
        assert _marker_condition("darwin", "true") is False


def _bf16_sites(filename: str):
    """Where ``torch.bfloat16`` appears in a test file, by test.

    Returns ``(by_test, strays)``: ``by_test`` maps each test that lists the dtype in a
    ``parametrize`` to the number of times it does; ``strays`` counts the occurrences
    outside any ``parametrize`` decorator (a test body that builds a bf16 head by hand
    is not seen by the fixture check below).
    """
    tree = ast.parse((_TESTS_DIR / filename).read_text(encoding="utf-8"))
    every = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "bfloat16"
    ]
    by_test: dict[str, int] = {}
    in_parametrize = set()
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        for decorator in func.decorator_list:
            if not _is_call_to(decorator, "parametrize"):
                continue
            for node in ast.walk(decorator):
                if isinstance(node, ast.Attribute) and node.attr == "bfloat16":
                    in_parametrize.add(id(node))
                    by_test[func.name] = by_test.get(func.name, 0) + 1
    strays = sum(1 for node in every if id(node) not in in_parametrize)
    return by_test, strays


class TestEveryBf16CaseOfTheLossChunkTestsRequestsTheFixture:
    """Run 40014485149 died on the first bf16 case, and every other one is the same matmul.

    ``FIXTURE_TESTS`` pins the three tests that list bfloat16 today. This reads the file
    the other way round, so that a FOURTH test with a bfloat16 parameter, or a bf16 head
    built inside a test body, cannot arrive without the fixture and bring the Windows
    cell down again.
    """

    FILE = "test_stream_loss_chunk.py"

    def test_each_test_with_a_bfloat16_parameter_requests_the_fixture(self):
        by_test, _ = _bf16_sites(self.FILE)
        missing = sorted(
            name
            for name in by_test
            if "aten_half_matmuls" not in _param_names(_find_test(self.FILE, name))
        )
        assert not missing, (
            f"{self.FILE}: {missing} list bfloat16 without requesting aten_half_matmuls, "
            f"so their CPU bf16 matmul reaches oneDNN on every Windows CI runner (#382)"
        )

    def test_exactly_the_pinned_tests_list_bfloat16(self):
        by_test, _ = _bf16_sites(self.FILE)
        assert set(by_test) == set(FIXTURE_TESTS[self.FILE]), (
            f"{self.FILE}: tests with a bfloat16 parameter {sorted(by_test)} differ from the "
            f"pinned {sorted(FIXTURE_TESTS[self.FILE])}; add the new one to FIXTURE_TESTS"
        )
        assert all(count == 1 for count in by_test.values()), by_test

    def test_no_bfloat16_outside_a_parametrize(self):
        _, strays = _bf16_sites(self.FILE)
        assert strays == 0, (
            f"{self.FILE}: {strays} torch.bfloat16 outside a parametrize is not seen by "
            f"the fixture check; parametrize it and request aten_half_matmuls"
        )


class TestTheHalfPrecisionTestsRequestTheFixture:
    """Each test that swapped the skip for ``aten_half_matmuls`` must keep it."""

    @pytest.mark.parametrize(
        "filename, target",
        [(f, t) for f, targets in FIXTURE_TESTS.items() for t in targets],
    )
    def test_the_test_requests_the_fixture(self, filename, target):
        node = _find_test(filename, target)
        assert "aten_half_matmuls" in _param_names(node), (
            f"{filename}::{target} runs a CPU bf16/fp16 step and must request "
            f"aten_half_matmuls (#1314)"
        )
        assert "skip_on_windows_ci" not in _decorator_names(node), (
            f"{filename}::{target} requests aten_half_matmuls and must not also skip; "
            f"if it needs the #382 skip back, move it to GUARDED_TESTS and link the log"
        )

@pytest.mark.skipif(
    os.environ.get("CI") == "true" and sys.platform == "win32",
    reason="the control below asserts the unskipped state",
)
def test_the_guard_does_not_skip_anywhere_else():
    """Everywhere but a Windows CI runner, the marker must be inert."""
    assert _windows_ci() is False
