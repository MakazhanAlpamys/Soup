"""#379: publish the STEP 14 control-versus-repair measurement harness.

The historical gate compared a streamed NF4 control and the repaired streamed
path against one resident NF4 reference. Its first run reported ``0/0`` exact
gradients because the parameter-name intersection was empty. These tests pin
the non-vacuous contract on CPU; real 32B/72B numbers still require CUDA.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_DIR = _REPO_ROOT / "benchmarks" / "harness"
_HARNESS = _HARNESS_DIR / "variant2_gate.py"


def _load_harness():
    sys.path.insert(0, str(_HARNESS_DIR))
    try:
        spec = importlib.util.spec_from_file_location("variant2_gate", _HARNESS)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_HARNESS_DIR))


_harness = _load_harness()


class _FakeParameter:
    def __init__(self, grad=object()) -> None:
        self.grad = grad


class _FakeModel:
    def __init__(self, names: list[str], *, grad=object()) -> None:
        self._parameters = [(name, _FakeParameter(grad)) for name in names]

    def named_parameters(self):
        return iter(self._parameters)


class TestArgumentValidation:
    @pytest.mark.parametrize(
        ("flag", "value"),
        [("--seq", "0"), ("--repeats", "0"), ("--buffers", "1")],
    )
    def test_bad_protocol_values_fail_before_cuda_skip(
        self, monkeypatch, capsys, flag: str, value: str
    ) -> None:
        monkeypatch.setattr(_harness, "cuda_available", lambda: False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["variant2_gate.py", "--weights", "x", "--shards", "y", flag, value],
        )

        assert _harness.main() == 2
        assert flag in capsys.readouterr().out

    def test_valid_protocol_is_an_intentional_no_cuda_skip(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(_harness, "cuda_available", lambda: False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["variant2_gate.py", "--weights", "x", "--shards", "y"],
        )

        assert _harness.main() == 0
        assert "intentional skip" in capsys.readouterr().out.lower()


class TestCanonicalGradientIntersection:
    def test_empty_model_intersection_is_a_hard_failure(self) -> None:
        left = _FakeModel(["model.layers.0.q_proj.lora_A.default.weight"])
        right = _FakeModel(["model.layers.1.q_proj.lora_A.default.weight"])

        with pytest.raises(ValueError, match="no canonical parameter name"):
            _harness.shared_lora_gradient_names(left, right)

    def test_shared_parameters_without_shared_lora_gradients_fail(self) -> None:
        left = _FakeModel(["model.embed_tokens.weight"])
        right = _FakeModel(["model.embed_tokens.weight"])

        with pytest.raises(ValueError, match="no shared LoRA gradient"):
            _harness.shared_lora_gradient_names(left, right)

    def test_shared_lora_gradient_names_are_returned(self) -> None:
        name = "model.layers.0.q_proj.lora_A.default.weight"
        assert _harness.shared_lora_gradient_names(
            _FakeModel([name]), _FakeModel([name])
        ) == frozenset({name})


class TestGradientVerdict:
    def test_gate_requires_the_control_to_break_and_repair_to_be_exact(self) -> None:
        broken = _harness.GradientSummary(compared=4, exact=1, wrong_layers=2, worst_abs=0.5)
        exact = _harness.GradientSummary(compared=4, exact=4, wrong_layers=0, worst_abs=0.0)

        assert _harness.gate_passes(control=broken, repaired=exact)
        assert not _harness.gate_passes(control=exact, repaired=exact)
        assert not _harness.gate_passes(control=broken, repaired=broken)

    def test_every_repetition_must_have_a_matching_control_and_repair(self) -> None:
        broken = _harness.GradientSummary(compared=4, exact=1, wrong_layers=2, worst_abs=0.5)
        exact = _harness.GradientSummary(compared=4, exact=4, wrong_layers=0, worst_abs=0.0)

        assert _harness._paired_gradient_gate(
            {"gradients": [vars(broken), vars(broken)]},
            {"gradients": [vars(exact), vars(exact)]},
        )
        assert not _harness._paired_gradient_gate(
            {"gradients": [vars(broken), vars(broken)]},
            {"gradients": [vars(exact)]},
        )

    def test_real_tensors_distinguish_exact_and_broken_gradients(self) -> None:
        torch = pytest.importorskip("torch")
        name = "model.layers.3.q_proj.lora_A.default.weight"
        left = _FakeModel([name], grad=torch.tensor([1.0, 2.0]))
        same = _FakeModel([name], grad=torch.tensor([1.0, 2.0]))
        different = _FakeModel([name], grad=torch.tensor([1.0, 4.0]))

        exact = _harness.compare_lora_gradients(left, same)
        broken = _harness.compare_lora_gradients(left, different)

        assert exact == _harness.GradientSummary(1, 1, 0, 0.0)
        assert broken == _harness.GradientSummary(1, 0, 1, 2.0)


def _gate_result(
    *, exact: int, wrong_layers: int, rewired_modules: int, losses_exact: bool = True
) -> dict[str, Any]:
    return {
        "gradients": [vars(_harness.GradientSummary(256, exact, wrong_layers, 0.0))],
        "losses_exact": losses_exact,
        "rewired_modules": rewired_modules,
    }


class TestWholeRunVerdict:
    @pytest.mark.parametrize(
        ("control", "repaired", "expected"),
        [
            ((8, 62, 0, True), (256, 0, 448, True), (True, True, True, True, True)),
            ((256, 0, 0, True), (256, 0, 0, True), (False, False, True, True, False)),
            ((8, 62, 0, True), (256, 0, 0, True), (False, True, True, True, False)),
            ((8, 62, 7, True), (256, 0, 448, True), (False, True, True, True, False)),
            ((8, 62, 0, False), (256, 0, 448, True), (False, False, True, False, True)),
            ((8, 62, 0, True), (256, 0, 448, False), (False, True, False, False, True)),
        ],
    )
    def test_verdict_pins_gradients_losses_and_wiring(
        self,
        control: tuple[int, int, int, bool],
        repaired: tuple[int, int, int, bool],
        expected: tuple[bool, bool, bool, bool, bool],
    ) -> None:
        control_result = _gate_result(
            exact=control[0],
            wrong_layers=control[1],
            rewired_modules=control[2],
            losses_exact=control[3],
        )
        repaired_result = _gate_result(
            exact=repaired[0],
            wrong_layers=repaired[1],
            rewired_modules=repaired[2],
            losses_exact=repaired[3],
        )

        assert _harness.verdict(control_result, repaired_result) == expected

    @pytest.mark.parametrize(
        ("control", "repaired", "exit_code"),
        [
            ((8, 62, 0, True), (256, 0, 448, True), 0),
            ((256, 0, 0, True), (256, 0, 0, True), 1),
            ((8, 62, 0, True), (256, 0, 0, True), 1),
            ((8, 62, 7, True), (256, 0, 448, True), 1),
            ((8, 62, 0, False), (256, 0, 448, True), 1),
            ((8, 62, 0, True), (256, 0, 448, False), 1),
        ],
    )
    def test_main_builds_a_real_control_and_reports_the_verdict(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        control: tuple[int, int, int, bool],
        repaired: tuple[int, int, int, bool],
        exit_code: int,
    ) -> None:
        torch = pytest.importorskip("torch")
        from soup_cli.utils import layer_shard, layer_stream_runtime, spectrum_scan

        monkeypatch.setattr(
            _harness,
            "parse_args",
            lambda: SimpleNamespace(
                weights="fake",
                shards=str(tmp_path / "shards"),
                seq=128,
                batch=1,
                buffers=2,
                repeats=1,
                json=None,
            ),
        )
        monkeypatch.setattr(_harness, "cuda_available", lambda: True)
        monkeypatch.setattr(_harness, "_source_sha", lambda: "a" * 40)
        monkeypatch.setattr(_harness, "_versions", lambda: {"commit": "a" * 40})
        monkeypatch.setattr(layer_stream_runtime, "build_meta_skeleton", lambda *a, **k: object())
        monkeypatch.setattr(layer_stream_runtime, "quantised_layer_suffixes", lambda *_: ())
        monkeypatch.setattr(layer_shard, "shard_checkpoint", lambda *a, **k: object())
        monkeypatch.setattr(spectrum_scan, "resolve_model_weights", lambda *_: "fake-weights")
        monkeypatch.setattr(_harness.shared, "model_arch_name", lambda *_: "fake-arch")
        monkeypatch.setattr(
            _harness.shared,
            "load_resident_reference",
            lambda *a, **k: SimpleNamespace(config=SimpleNamespace(vocab_size=16)),
        )
        monkeypatch.setattr(_harness.shared, "make_non_vacuous_lora", lambda *_: None)
        monkeypatch.setattr(_harness.shared, "copy_lora", lambda *a: None)
        monkeypatch.setattr(
            torch, "Generator", lambda **k: SimpleNamespace(manual_seed=lambda *_: object())
        )
        monkeypatch.setattr(torch, "randint", lambda *a, **k: object())
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "simulated CUDA")
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

        build_flags: list[bool] = []

        def fake_build_arm(*, repair_enabled: bool, **kwargs: object) -> tuple[object, Any]:
            build_flags.append(repair_enabled)
            return object(), SimpleNamespace(close=lambda: None)

        results = {
            "control": _gate_result(
                exact=control[0],
                wrong_layers=control[1],
                rewired_modules=control[2],
                losses_exact=control[3],
            ),
            "repaired": _gate_result(
                exact=repaired[0],
                wrong_layers=repaired[1],
                rewired_modules=repaired[2],
                losses_exact=repaired[3],
            ),
        }

        def fake_run_arm(*, label: str, **kwargs: object) -> dict[str, Any]:
            return results[label]

        monkeypatch.setattr(_harness, "_build_arm", fake_build_arm)
        monkeypatch.setattr(_harness, "_run_arm", fake_run_arm)

        assert _harness.main() == exit_code
        assert build_flags == [False, True]
        assert f"RESULT: {'passed' if exit_code == 0 else 'failed'}" in capsys.readouterr().out


class TestSourceCommit:
    @pytest.mark.parametrize("status", ["", " M src/soup_cli/changed.py\n"])
    def test_source_sha_uses_imported_package_tree_and_marks_dirty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str
    ) -> None:
        import soup_cli

        package_file = soup_cli.__file__
        assert package_file is not None
        package_dir = Path(package_file).resolve().parent
        expected_sha = "a" * 40
        calls: list[tuple[list[str], Path]] = []

        def fake_run(command: list[str], *, cwd: Path, **kwargs: object) -> Any:
            calls.append((command, cwd))
            if command == ["git", "rev-parse", "--show-toplevel"]:
                return SimpleNamespace(stdout=f"{tmp_path}\n")
            if command == ["git", "rev-parse", "HEAD"]:
                return SimpleNamespace(stdout=f"{expected_sha}\n")
            return SimpleNamespace(stdout=status)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(_harness.subprocess, "run", fake_run)

        assert _harness._source_sha() == expected_sha + ("-dirty" if status else "")
        assert calls == [
            (["git", "rev-parse", "--show-toplevel"], package_dir),
            (["git", "rev-parse", "HEAD"], tmp_path),
            (["git", "status", "--porcelain", "--untracked-files=normal"], tmp_path),
        ]


class TestRepairToggle:
    def test_control_disables_repair_only_while_building(self) -> None:
        calls: list[str] = []

        def install(_module) -> int:
            calls.append("repair")
            return 7

        runtime_module = SimpleNamespace(install_dequant_forward=install)

        def build_model():
            return runtime_module.install_dequant_forward(object())

        assert (
            _harness.build_streamed_arm(
                repair_enabled=False,
                runtime_module=runtime_module,
                build_model=build_model,
            )
            == 0
        )
        assert calls == []
        assert runtime_module.install_dequant_forward is install

        assert (
            _harness.build_streamed_arm(
                repair_enabled=True,
                runtime_module=runtime_module,
                build_model=build_model,
            )
            == 7
        )
        assert calls == ["repair"]


class TestRewiringCounter:
    def test_real_torch_wrapper_hidden_by_named_modules_is_counted(self) -> None:
        torch = pytest.importorskip("torch")

        class HiddenWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = torch.nn.Identity()
                self.n_dequant_forward = 7

            def named_modules(self, memo=None, prefix="", remove_duplicate=True):
                yield from self.inner.named_modules(memo, prefix, remove_duplicate)

        wrapper = HiddenWrapper()
        model = torch.nn.Sequential(wrapper)
        assert all(module is not wrapper for module in model.modules())
        assert _harness._rewired_modules(model) == 7

    def test_walks_structural_children_hidden_by_named_modules(self) -> None:
        inner = SimpleNamespace(_modules={})
        wrapper = SimpleNamespace(_modules={"inner": inner}, n_dequant_forward=7)
        root = SimpleNamespace(_modules={"first": wrapper, "alias": wrapper})
        # Model.modules() delegates to the overridden named_modules() and
        # hides StreamedDecoderLayer itself after #1010.
        root.modules = lambda: iter((root, inner))

        assert _harness._rewired_modules(root) == 7

    def test_repaired_arm_fails_loudly_before_measurement_when_counter_is_dead(self) -> None:
        model = SimpleNamespace(_modules={}, train=lambda: pytest.fail("entered training"))

        with pytest.raises(RuntimeError, match="repaired arm.*no rewired modules"):
            _harness._run_arm(
                label="repaired",
                model=model,
                reference=object(),
                input_ids=object(),
                repeats=1,
            )


def test_harness_is_indexed() -> None:
    readme = (_REPO_ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    assert "harness/variant2_gate.py" in readme
