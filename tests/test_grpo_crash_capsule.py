"""CPU safety and serialization tests for the opt-in GRPO crash capsule."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks" / "harness" / "grpo_crash_capsule.py"


def _module():
    name = "grpo_crash_capsule_test_module"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_harness_keeps_torch_import_lazy():
    source = HARNESS.read_text(encoding="utf-8")

    assert "\nimport torch\n" not in source
    assert "def _torch()" in source


def test_writer_requires_a_new_user_named_directory_and_never_overwrites(tmp_path):
    harness = _module()
    target = tmp_path / "issue342-capture"
    writer = harness.CapsuleWriter.create(target, max_total_bytes=512)

    assert writer.directory == target / "rank000"
    writer.write_json("manifest.json", {"schema": 1})
    assert json.loads((writer.directory / "manifest.json").read_text(encoding="utf-8")) == {
        "schema": 1
    }
    with pytest.raises(harness.CapsuleError, match="one relative component"):
        writer.write_json("../outside.json", {"no": "write"})
    with pytest.raises(harness.CapsuleError, match="overwrite capsule file"):
        writer.write_json("manifest.json", {"schema": 2})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        harness.CapsuleWriter.create(target, max_total_bytes=512)


def test_writer_enforces_its_per_rank_cap_before_creating_the_file(tmp_path):
    harness = _module()
    writer = harness.CapsuleWriter.create(tmp_path / "bounded", max_total_bytes=48)

    with pytest.raises(harness.CapsuleSizeLimitError, match="cap"):
        writer.write_json("too-large.json", {"row": "x" * 100})
    assert list(writer.directory.iterdir()) == []


def test_writer_refuses_a_global_cap_smaller_than_the_world_size(tmp_path):
    harness = _module()

    with pytest.raises(ValueError, match="at least world_size"):
        harness.CapsuleWriter.create(
            tmp_path / "too-small-for-ranks", max_total_bytes=1, rank=0, world_size=2
        )

    assert not (tmp_path / "too-small-for-ranks").exists()


def test_multi_rank_writer_validates_distributed_support_before_creating_a_root(tmp_path):
    harness = _module()
    target = tmp_path / "needs-distributed"

    with pytest.raises(harness.CapsuleError, match="requires torch_module"):
        harness.CapsuleWriter.create(target, max_total_bytes=48, world_size=2)

    assert not target.exists()


def test_multi_rank_writer_broadcasts_rank_zero_creation_failure_before_barrier(tmp_path):
    harness = _module()
    target = tmp_path / "already-exists"
    target.mkdir()

    class Distributed:
        broadcast_calls = []
        barrier_calls = 0

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @classmethod
        def broadcast_object_list(cls, objects, src):
            cls.broadcast_calls.append((list(objects), src))

        @classmethod
        def barrier(cls):
            cls.barrier_calls += 1

    fake_torch = SimpleNamespace(distributed=Distributed)
    with pytest.raises(harness.CapsuleError, match="rank zero could not create"):
        harness.CapsuleWriter.create(
            target,
            max_total_bytes=48,
            rank=0,
            world_size=2,
            torch_module=fake_torch,
        )

    assert Distributed.broadcast_calls[0][1] == 0
    assert Distributed.barrier_calls == 0


def test_capsule_requires_explicit_sensitive_row_acknowledgement(tmp_path):
    harness = _module()

    with pytest.raises(harness.CapsuleError, match="acknowledge_training_rows=True"):
        harness.Issue342CrashCapsule(
            SimpleNamespace(model=object()), output_dir=tmp_path / "not-created"
        )
    assert not (tmp_path / "not-created").exists()


def test_anomaly_capsule_is_weights_only_loadable_and_precedes_a_watchdog(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(1, 1, bias=False)

    class Trainer:
        def __init__(self):
            self.model = Model()
            self.state = SimpleNamespace(global_step=7)
            self.callback_handler = SimpleNamespace(callbacks=[Watchdog(self.model)])

        def add_callback(self, callback):
            self.callback_handler.callbacks.append(callback)

        def _compute_loss(self, model, inputs):
            return (model.lora_A(inputs["input_ids"]) * inputs["poison"]).sum()

    class Watchdog:
        def __init__(self, model):
            self.model = model

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            del args, state, kwargs
            for parameter in self.model.parameters():
                parameter.grad = None
            return control

    trainer = Trainer()
    harness.Issue342CrashCapsule(
        trainer,
        output_dir=tmp_path / "capsule",
        max_total_bytes=1 << 20,
        acknowledge_training_rows=True,
    ).install()
    assert type(trainer.callback_handler.callbacks[0]).__name__ != "Watchdog"

    inputs = {
        "input_ids": torch.ones((1, 1), requires_grad=True),
        "poison": torch.tensor([[float("nan")]]),
    }
    loss = trainer._compute_loss(trainer.model, inputs)
    loss.backward()
    for callback in trainer.callback_handler.callbacks:
        callback.on_pre_optimizer_step(None, trainer.state, None, model=trainer.model)
    assert trainer.model.lora_A.weight.grad is None

    rank_dir = tmp_path / "capsule" / "rank000"
    payload_path = rank_dir / "capsule_000001.pt"
    summary_path = rank_dir / "capsule_000001.json"
    assert payload_path.is_file()
    payload = torch.load(payload_path, weights_only=True)
    assert torch.equal(payload["microbatches"][0]["input_ids"], inputs["input_ids"])
    assert "lora_A.weight" in payload["post_clip_nonfinite_gradient_names"]
    assert payload["lora_observations"]["lora_A"][0]["input"]["finite"] is True
    assert payload["first_nonfinite_activation"]["stage"] == "grad_output"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["reason"] == "nonfinite_before_watchdog"
    assert summary["post_clip_nonfinite_gradient_names"] == ["lora_A.weight"]


def test_a_finite_step_leaves_no_training_rows_on_disk(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    class Trainer:
        def __init__(self):
            self.model = Model()
            self.state = SimpleNamespace(global_step=0)
            self.callback_handler = SimpleNamespace(callbacks=[])

        def add_callback(self, callback):
            self.callback_handler.callbacks.append(callback)

        def _compute_loss(self, model, inputs):
            return (model.weight * inputs["input_ids"]).sum()

    trainer = Trainer()
    original_loss = trainer._compute_loss
    capsule = harness.Issue342CrashCapsule(
        trainer,
        output_dir=tmp_path / "normal",
        max_total_bytes=1 << 20,
        acknowledge_training_rows=True,
        instrument_lora_operands=False,
    ).install()
    trainer._compute_loss(trainer.model, {"input_ids": torch.ones(1)}).backward()
    callback = trainer.callback_handler.callbacks[0]
    callback.on_pre_optimizer_step(None, trainer.state, None)
    callback.on_step_end(None, trainer.state, None)

    names = sorted(path.name for path in (tmp_path / "normal" / "rank000").iterdir())
    assert names == ["capture_policy.json", "manifest.json"]
    capsule.close()
    assert trainer._compute_loss == original_loss
    assert trainer.callback_handler.callbacks == []


def _diagnostic_trainer(torch):
    class Trainer:
        def __init__(self):
            self.model = torch.nn.Linear(16, 1, bias=False)
            with torch.no_grad():
                self.model.weight.zero_()
            self.state = SimpleNamespace(global_step=0)
            self.callback_handler = SimpleNamespace(callbacks=[])
            self.accelerator = SimpleNamespace(clip_grad_norm_=torch.nn.utils.clip_grad_norm_)

        def add_callback(self, callback):
            self.callback_handler.callbacks.append(callback)

        def _compute_loss(self, model, inputs):
            return model(inputs["input_ids"]).sum() + inputs["constant"]

    return Trainer()


@pytest.mark.parametrize("bad_index", [0, 1])
def test_capsule_preserves_each_microbatch_loss_and_first_bad_loss(tmp_path, bad_index):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "losses",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    for index in range(2):
        trainer._compute_loss(trainer.model, {
            "input_ids": torch.ones(1, 16),
            "constant": torch.tensor(float("nan") if index == bad_index else 0.0),
        }).backward()
    capsule.capture_pre_optimizer(trainer.state)
    payload = torch.load(tmp_path / "losses/rank000/capsule_000001.pt", weights_only=True)
    assert payload["loss"]["finite"] is (bad_index == 0)
    assert [row["microbatch_index"] for row in payload["microbatch_loss_stats"]] == [0, 1]
    assert [row["stats"]["finite"] for row in payload["microbatch_loss_stats"]] == [
        bad_index != 0, bad_index != 1,
    ]
    summary = json.loads(
        (tmp_path / "losses/rank000/capsule_000001.json").read_text(encoding="utf-8")
    )
    assert summary["first_nonfinite_loss"]["microbatch_index"] == bad_index
    capsule.close()


def test_first_nonfinite_loss_is_preserved_when_both_microbatches_are_bad(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "two-bad-losses",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    for _ in range(2):
        trainer._compute_loss(trainer.model, {
            "input_ids": torch.ones(1, 16), "constant": torch.tensor(float("nan")),
        }).backward()
    capsule.capture_pre_optimizer(trainer.state)
    payload = torch.load(
        tmp_path / "two-bad-losses/rank000/capsule_000001.pt", weights_only=True,
    )
    assert [row["microbatch_index"] for row in payload["microbatch_loss_stats"]] == [0, 1]
    assert [row["stats"]["finite"] for row in payload["microbatch_loss_stats"]] == [False, False]
    assert payload["first_nonfinite_loss"]["microbatch_index"] == 0
    capsule.close()


def test_capsule_observes_accumulation_overflow_from_finite_contributions(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    contributions = []
    trainer.model.weight.register_hook(
        lambda grad: contributions.append(bool(grad.isfinite().all()))
    )
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "accumulation",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    for _ in range(2):
        trainer._compute_loss(trainer.model, {
            "input_ids": torch.full((1, 16), 2e38), "constant": torch.tensor(0.0),
        }).backward()
    assert contributions == [True, True]
    assert trainer.model.weight.grad.isinf().all()
    capsule.capture_pre_optimizer(trainer.state)
    summary = json.loads(
        (tmp_path / "accumulation/rank000/capsule_000001.json").read_text(encoding="utf-8")
    )
    assert summary["first_nonfinite_parameter"] is None
    assert summary["nonfinite_parameter_names_before_clip"] == []
    assert summary["first_nonfinite_accumulated_parameter"]["name"] == "weight"
    assert summary["first_nonfinite_accumulated_parameter"]["microbatch_index"] == 1
    assert summary["nonfinite_accumulated_parameter_names_before_clip"] == ["weight"]
    capsule.close()
    assert not trainer.model.weight._post_accumulate_grad_hooks


def test_accumulation_attribution_keeps_first_overflow_and_unique_names(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    contributions = []
    trainer.model.weight.register_hook(
        lambda grad: contributions.append(bool(grad.isfinite().all()))
    )
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "three-backwards",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    for _ in range(3):
        trainer._compute_loss(trainer.model, {
            "input_ids": torch.full((1, 16), 2e38), "constant": torch.tensor(0.0),
        }).backward()
    assert contributions == [True, True, True]
    assert trainer.model.weight.grad.isinf().all()
    capsule.capture_pre_optimizer(trainer.state)
    summary = json.loads(
        (tmp_path / "three-backwards/rank000/capsule_000001.json").read_text(encoding="utf-8")
    )
    assert summary["first_nonfinite_parameter"] is None
    assert summary["nonfinite_parameter_names_before_clip"] == []
    assert summary["first_nonfinite_accumulated_parameter"]["name"] == "weight"
    assert summary["first_nonfinite_accumulated_parameter"]["microbatch_index"] == 1
    assert summary["nonfinite_accumulated_parameter_names_before_clip"] == ["weight"]
    capsule.close()


def test_finite_returned_norm_preserves_an_earlier_loss_anomaly(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "bad-loss-finite-norm",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    trainer._compute_loss(trainer.model, {
        "input_ids": torch.ones(1, 16), "constant": torch.tensor(float("nan")),
    }).backward()
    assert trainer.model.weight.grad.isfinite().all()
    norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1.0)
    assert norm.item() == 4.0
    assert trainer.model.weight.grad.isfinite().all()
    capsule.capture_pre_optimizer(trainer.state)
    summary_path = tmp_path / "bad-loss-finite-norm/rank000/capsule_000001.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["first_nonfinite_loss"]["microbatch_index"] == 0
    assert summary["gradient_norm_observation"]["finite"] is True
    assert summary["gradient_norm_observation"]["value"] == 4.0
    assert summary["post_clip_nonfinite_gradient_names"] == []
    capsule.close()


@pytest.mark.parametrize("max_norm", [1.0, float("inf")])
def test_capsule_triggers_on_the_actual_nonfinite_norm_and_restores_the_method(tmp_path, max_norm):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    original = trainer.accelerator.clip_grad_norm_
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "norm",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    trainer._compute_loss(trainer.model, {
        "input_ids": torch.full((1, 16), 1e20), "constant": torch.tensor(0.0),
    }).backward()
    assert trainer.model.weight.grad.isfinite().all()
    norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), max_norm)
    assert norm.isinf()
    capsule.capture_pre_optimizer(trainer.state)
    summary_path = tmp_path / "norm/rank000/capsule_000001.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["gradient_norm_observation"]["finite"] is False
    assert summary["gradient_norm_observation"]["value"] == "inf"
    assert summary["gradient_norm_source"] == "accelerator.clip_grad_norm_"
    assert summary["first_nonfinite_parameter"] is None
    assert summary["first_nonfinite_accumulated_parameter"] is None
    if max_norm == 1.0:
        assert trainer.model.weight.grad.isfinite().all()
        assert summary["post_clip_nonfinite_gradient_names"] == []
    capsule.close()
    assert trainer.accelerator.clip_grad_norm_ is original


@pytest.mark.parametrize("hook_kind", ["missing", "none", "zero", "empty"])
def test_missing_accumulation_hook_is_explicitly_unavailable(tmp_path, monkeypatch, hook_kind):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    method = "register_post_accumulate_grad_hook"
    if hook_kind == "missing":
        monkeypatch.delattr(torch.Tensor, method)
    else:
        monkeypatch.setattr(torch.Tensor, method, {"none": None, "zero": 0, "empty": ""}[hook_kind])
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "unsupported-hook",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    trainer._compute_loss(trainer.model, {
        "input_ids": torch.ones(1, 16), "constant": torch.tensor(float("nan")),
    }).backward()
    assert trainer.model.weight.grad.isfinite().all()
    capsule.capture_pre_optimizer(trainer.state)
    payload = torch.load(
        tmp_path / "unsupported-hook/rank000/capsule_000001.pt", weights_only=True
    )
    manifest = json.loads(
        (tmp_path / "unsupported-hook/rank000/manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (tmp_path / "unsupported-hook/rank000/capsule_000001.json").read_text(encoding="utf-8")
    )
    for record in (manifest, payload["environment"], summary):
        assert record["accumulated_gradient_source"] == "unavailable"
        assert record["accumulated_gradient_unavailable_parameter_names"] == ["weight"]
    assert manifest["accumulated_gradient_hook_count"] == 0
    assert manifest["accumulated_gradient_phase"] == "unavailable"
    assert payload["first_nonfinite_accumulated_parameter"] is None
    capsule.close()


def test_partial_accumulation_hook_support_lists_unobserved_parameters(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    trainer.model.extra = torch.nn.Parameter(torch.zeros(1))
    monkeypatch.setattr(trainer.model.extra, "register_post_accumulate_grad_hook", None)
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "partial-hooks",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    for _ in range(2):
        trainer._compute_loss(trainer.model, {
            "input_ids": torch.full((1, 16), 2e38), "constant": torch.tensor(0.0),
        }).backward()
    capsule.capture_pre_optimizer(trainer.state)
    summary = json.loads(
        (tmp_path / "partial-hooks/rank000/capsule_000001.json").read_text(encoding="utf-8")
    )
    assert summary["accumulated_gradient_source"] == (
        "torch.Tensor.register_post_accumulate_grad_hook"
    )
    assert summary["accumulated_gradient_unavailable_parameter_names"] == ["extra"]
    assert summary["first_nonfinite_accumulated_parameter"]["name"] == "weight"
    assert summary["first_nonfinite_accumulated_parameter"]["microbatch_index"] == 1
    capsule.close()
    assert not trainer.model.weight._post_accumulate_grad_hooks


@pytest.mark.parametrize("path", ["no_accelerator", "no_method", "none_method"])
def test_missing_norm_method_leaves_its_source_unavailable(tmp_path, path):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    if path == "no_accelerator":
        del trainer.accelerator
    elif path == "no_method":
        trainer.accelerator = SimpleNamespace()
    else:
        trainer.accelerator.clip_grad_norm_ = None
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "unsupported-norm",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    trainer._compute_loss(trainer.model, {
        "input_ids": torch.ones(1, 16), "constant": torch.tensor(float("nan")),
    }).backward()
    capsule.capture_pre_optimizer(trainer.state)
    summary = json.loads(
        (tmp_path / "unsupported-norm/rank000/capsule_000001.json").read_text(encoding="utf-8")
    )
    assert summary["gradient_norm_source"] == "unavailable"
    assert summary["gradient_norm_observation"] is None
    capsule.close()
    if path == "no_accelerator":
        assert not hasattr(trainer, "accelerator")
    elif path == "no_method":
        assert not hasattr(trainer.accelerator, "clip_grad_norm_")
    else:
        assert trainer.accelerator.clip_grad_norm_ is None


@pytest.mark.parametrize("result_kind", ["none", "empty", "non_scalar", "complex", "huge_integer"])
def test_unsupported_norm_result_passes_through_without_a_false_observation(tmp_path, result_kind):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    result = {
        "none": None, "empty": "", "non_scalar": torch.ones(2),
        "complex": torch.tensor(1 + 2j), "huge_integer": 10**400,
    }[result_kind]
    calls = []

    def norm(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    trainer.accelerator.clip_grad_norm_ = norm
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "unsupported-result",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    trainer._compute_loss(trainer.model, {
        "input_ids": torch.ones(1, 16), "constant": torch.tensor(0.0),
    }).backward()
    parameters = list(trainer.model.parameters())
    assert trainer.accelerator.clip_grad_norm_(parameters, 1.0) is result
    assert calls == [((parameters, 1.0), {})]
    assert capsule._current["gradient_norm_source"] == "unavailable"
    assert capsule._current["gradient_norm_observation"] is None
    capsule.capture_pre_optimizer(trainer.state)
    assert not list((tmp_path / "unsupported-result/rank000").glob("capsule_*"))
    capsule.close()
    assert trainer.accelerator.clip_grad_norm_ is norm


def test_late_install_failure_removes_live_hooks_and_restores_methods(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    original_norm = trainer.accelerator.clip_grad_norm_
    original_loss = trainer._compute_loss
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "late-failure",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    )
    sentinel = RuntimeError("callback ordering refused")

    def fail(*args):
        assert trainer.model.weight._post_accumulate_grad_hooks
        assert trainer.accelerator.clip_grad_norm_ is not original_norm
        assert trainer._compute_loss != original_loss
        raise sentinel

    monkeypatch.setattr(capsule, "_place_before_watchdog", fail)
    with pytest.raises(RuntimeError) as error:
        capsule.install()
    assert error.value is sentinel
    assert trainer.accelerator.clip_grad_norm_ is original_norm
    assert trainer._compute_loss == original_loss
    assert not trainer.model.weight._post_accumulate_grad_hooks
    assert not trainer.model.weight._backward_hooks
    assert trainer.callback_handler.callbacks == []


def test_norm_observer_preserves_arguments_return_value_errors_and_step_reset(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    calls = []
    sentinel = RuntimeError("the norm implementation failed")
    result = torch.tensor(2.0)

    def norm(parameters, max_norm, *, norm_type=2.0):
        calls.append((parameters, max_norm, norm_type))
        if max_norm < 0:
            raise sentinel
        return result

    trainer.accelerator.clip_grad_norm_ = norm
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "observer",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    inputs = {"input_ids": torch.ones(1, 16), "constant": torch.tensor(0.0)}
    trainer._compute_loss(trainer.model, inputs).backward()
    parameters = trainer.model.parameters()
    assert trainer.accelerator.clip_grad_norm_(parameters, 1.0, norm_type=1.0) is result
    assert calls == [(parameters, 1.0, 1.0)]
    with pytest.raises(RuntimeError) as error:
        trainer.accelerator.clip_grad_norm_(parameters, -1.0)
    assert error.value is sentinel
    assert len(calls) == 2
    assert capsule._current["gradient_norm_observation"]["finite"]
    assert capsule._current["gradient_norm_observation"]["value"] == 2.0
    capsule.finish_step()
    trainer.model.zero_grad(set_to_none=True)
    trainer._compute_loss(trainer.model, inputs).backward()
    assert capsule._current["gradient_norm_source"] == "unavailable"
    assert capsule._current["gradient_norm_observation"] is None
    assert capsule._current["first_nonfinite_accumulated_parameter"] is None
    assert len(capsule._current["microbatch_loss_stats"]) == 1
    capsule.capture_pre_optimizer(trainer.state)
    assert not list((tmp_path / "observer/rank000").glob("capsule_*"))
    capsule.close()
    assert trainer.accelerator.clip_grad_norm_ is norm


def test_failed_install_restores_the_instance_norm_observer(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    original = trainer.accelerator.clip_grad_norm_
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "failed",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    )

    def fail(*args, **kwargs):
        raise OSError("writer refused the manifest")

    monkeypatch.setattr(capsule.writer, "write_json", fail)
    with pytest.raises(OSError, match="writer refused"):
        capsule.install()
    assert trainer.accelerator.clip_grad_norm_ is original
    assert not trainer.model.weight._post_accumulate_grad_hooks
    assert trainer.callback_handler.callbacks == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [(float("nan"), "nan"), (float("inf"), "inf"), (float("-inf"), "-inf")],
)
def test_norm_observer_classifies_python_nonfinite_scalars(tmp_path, value, expected):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    trainer.accelerator.clip_grad_norm_ = lambda *args, **kwargs: value
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "scalar",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    trainer._compute_loss(trainer.model, {
        "input_ids": torch.ones(1, 16), "constant": torch.tensor(0.0),
    }).backward()
    result = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1.0)
    assert result is value
    capsule.capture_pre_optimizer(trainer.state)
    summary = json.loads(
        (tmp_path / "scalar/rank000/capsule_000001.json").read_text(encoding="utf-8")
    )
    assert summary["gradient_norm_observation"]["value"] == expected
    capsule.close()


@pytest.mark.parametrize("value", [0.0, 1e200])
def test_finite_python_norm_preserves_zero_and_float64_range(tmp_path, value):
    torch = pytest.importorskip("torch")
    harness = _module()
    trainer = _diagnostic_trainer(torch)
    trainer.accelerator.clip_grad_norm_ = lambda *args, **kwargs: value
    capsule = harness.Issue342CrashCapsule(
        trainer, output_dir=tmp_path / "finite-norm",
        max_total_bytes=1 << 20, acknowledge_training_rows=True,
    ).install()
    trainer._compute_loss(trainer.model, {
        "input_ids": torch.ones(1, 16), "constant": torch.tensor(0.0),
    }).backward()
    assert trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1.0) is value
    stats = capsule._current["gradient_norm_observation"]
    assert stats["finite"] is True
    assert stats["dtype"] == "torch.float64"
    assert stats["value"] == value
    assert stats["abs_max"] == value
    capsule.capture_pre_optimizer(trainer.state)
    assert not list((tmp_path / "finite-norm/rank000").glob("capsule_*"))
    capsule.close()


@pytest.mark.parametrize(
    ("evaluation", "without_grad"),
    [(True, False), (False, True)],
    ids=["evaluation_with_grad", "training_without_grad"],
)
def test_nontraining_forward_does_not_open_a_training_capture_window(
    tmp_path, evaluation, without_grad
):
    torch = pytest.importorskip("torch")
    harness = _module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    class Trainer:
        def __init__(self):
            self.model = Model()
            self.state = SimpleNamespace(global_step=0)
            self.callback_handler = SimpleNamespace(callbacks=[])

        def add_callback(self, callback):
            self.callback_handler.callbacks.append(callback)

        def _compute_loss(self, model, inputs):
            return (model.weight * inputs["input_ids"]).sum()

    trainer = Trainer()
    capsule = harness.Issue342CrashCapsule(
        trainer,
        output_dir=tmp_path / "nontraining",
        max_total_bytes=1 << 20,
        acknowledge_training_rows=True,
        instrument_lora_operands=False,
    ).install()
    if evaluation:
        trainer.model.eval()
    if without_grad:
        with torch.no_grad():
            trainer._compute_loss(trainer.model, {"input_ids": torch.ones(1)})
    else:
        trainer._compute_loss(trainer.model, {"input_ids": torch.ones(1)})

    assert capsule._current is None
    assert sorted(path.name for path in capsule.writer.directory.iterdir()) == [
        "capture_policy.json",
        "manifest.json",
    ]


def test_capsule_distinguishes_pre_clip_attribution_from_post_clip_gradient_stats(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(1, 1, bias=False)
            self.lora_B = torch.nn.Linear(1, 1, bias=False)

    class Trainer:
        def __init__(self):
            self.model = Model()
            self.state = SimpleNamespace(global_step=0)
            self.callback_handler = SimpleNamespace(callbacks=[])

        def add_callback(self, callback):
            self.callback_handler.callbacks.append(callback)

        def _compute_loss(self, model, inputs):
            poisoned = model.lora_A(inputs["input_ids"]) * inputs["poison"]
            finite = model.lora_B(inputs["input_ids"])
            return poisoned.sum() + finite.sum()

    trainer = Trainer()
    harness.Issue342CrashCapsule(
        trainer,
        output_dir=tmp_path / "post-clip",
        max_total_bytes=1 << 20,
        acknowledge_training_rows=True,
        instrument_lora_operands=False,
    ).install()
    inputs = {
        "input_ids": torch.ones((1, 1)),
        "poison": torch.tensor([[float("nan")]]),
    }
    trainer._compute_loss(trainer.model, inputs).backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
    trainer.callback_handler.callbacks[0].on_pre_optimizer_step(None, trainer.state, None)

    payload = torch.load(
        tmp_path / "post-clip" / "rank000" / "capsule_000001.pt", weights_only=True
    )
    assert payload["first_nonfinite_parameter"]["name"] == "lora_A.weight"
    assert payload["nonfinite_parameter_names_before_clip"] == ["lora_A.weight"]
    assert sorted(payload["post_clip_nonfinite_gradient_names"]) == [
        "lora_A.weight",
        "lora_B.weight",
    ]
    assert payload["post_clip_pre_watchdog_gradient_stats"]["lora_B.weight"]["finite"] is False


def test_install_refuses_liger_loss_path_that_bypasses_the_wrapped_loss(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()

    class Trainer:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            self.args = SimpleNamespace(use_liger_kernel=True)
            self.state = SimpleNamespace(global_step=0)
            self.callback_handler = SimpleNamespace(callbacks=[])

        def add_callback(self, callback):
            self.callback_handler.callbacks.append(callback)

        def _compute_loss(self, model, inputs):
            return model(inputs["input_ids"]).sum()

    capsule = harness.Issue342CrashCapsule(
        Trainer(),
        output_dir=tmp_path / "liger",
        max_total_bytes=1 << 20,
        acknowledge_training_rows=True,
        instrument_lora_operands=False,
    )

    with pytest.raises(harness.CapsuleError, match="Liger"):
        capsule.install()


def test_install_refuses_a_rank_cap_smaller_than_the_pre_step_trainable_state(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()

    class Trainer:
        def __init__(self):
            self.model = torch.nn.Linear(64, 64, bias=False)
            self.state = SimpleNamespace(global_step=0)
            self.callback_handler = SimpleNamespace(callbacks=[])

        def add_callback(self, callback):
            self.callback_handler.callbacks.append(callback)

        def _compute_loss(self, model, inputs):
            return model(inputs["input_ids"]).sum()

    capsule = harness.Issue342CrashCapsule(
        Trainer(),
        output_dir=tmp_path / "too-small",
        max_total_bytes=1024,
        acknowledge_training_rows=True,
        instrument_lora_operands=False,
    )

    with pytest.raises(harness.CapsuleSizeLimitError, match="trainable snapshot"):
        capsule.install()


def test_install_projects_fp16_lora_alignment_for_cap_preflight(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(1024, 1024, bias=False, dtype=torch.bfloat16)

    class Trainer:
        def __init__(self):
            self.model = Model()
            self.args = SimpleNamespace(fp16=True, bf16=False)
            self.state = SimpleNamespace(global_step=0)
            self.callback_handler = SimpleNamespace(callbacks=[])

        def add_callback(self, callback):
            self.callback_handler.callbacks.append(callback)

        def _compute_loss(self, model, inputs):
            return model(inputs["input_ids"]).sum()

    trainer = Trainer()
    raw_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    capsule = harness.Issue342CrashCapsule(
        trainer,
        output_dir=tmp_path / "fp16-alignment",
        max_total_bytes=2_300_000,
        acknowledge_training_rows=True,
        instrument_lora_operands=False,
    )

    assert raw_bytes < capsule.writer.max_bytes
    with pytest.raises(harness.CapsuleSizeLimitError, match="FP16 LoRA alignment"):
        capsule.install()


def test_size_limited_payload_writes_a_summary_with_the_cap_error(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(1, 1, bias=False)

    class Trainer:
        def __init__(self):
            self.model = Model()
            self.state = SimpleNamespace(global_step=0)
            self.callback_handler = SimpleNamespace(callbacks=[])

        def add_callback(self, callback):
            self.callback_handler.callbacks.append(callback)

        def _compute_loss(self, model, inputs):
            return (model.lora_A(inputs["input_ids"]) * inputs["poison"]).sum()

    trainer = Trainer()
    capsule = harness.Issue342CrashCapsule(
        trainer,
        output_dir=tmp_path / "size-limited",
        max_total_bytes=8192,
        acknowledge_training_rows=True,
        instrument_lora_operands=False,
    ).install()
    # The tiny configured rank cap admits the compact JSON summary but not the
    # RNG/row/state Torch payload, exercising the recovery branch directly.
    inputs = {
        "input_ids": torch.ones((1, 1)),
        "poison": torch.tensor([[float("nan")]]),
    }
    trainer._compute_loss(trainer.model, inputs).backward()
    trainer.callback_handler.callbacks[0].on_pre_optimizer_step(None, trainer.state, None)

    rank_dir = capsule.writer.directory
    assert not (rank_dir / "capsule_000001.pt").exists()
    summary = json.loads((rank_dir / "capsule_000001.json").read_text(encoding="utf-8"))
    assert "rank cap" in summary["error"]


def test_size_limited_payload_warns_when_its_summary_cannot_fit(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(1, 1, bias=False)

    class Trainer:
        def __init__(self):
            self.model = Model()
            self.state = SimpleNamespace(global_step=0)
            self.callback_handler = SimpleNamespace(callbacks=[])

        def add_callback(self, callback):
            self.callback_handler.callbacks.append(callback)

        def _compute_loss(self, model, inputs):
            return (model.lora_A(inputs["input_ids"]) * inputs["poison"]).sum()

    trainer = Trainer()
    capsule = harness.Issue342CrashCapsule(
        trainer,
        output_dir=tmp_path / "summary-too-large",
        max_total_bytes=1 << 20,
        acknowledge_training_rows=True,
        instrument_lora_operands=False,
    ).install()
    capsule.writer.max_bytes = capsule.writer.bytes_written
    inputs = {
        "input_ids": torch.ones((1, 1)),
        "poison": torch.tensor([[float("nan")]]),
    }
    trainer._compute_loss(trainer.model, inputs).backward()

    with pytest.warns(RuntimeWarning, match="summary could not be written"):
        trainer.callback_handler.callbacks[0].on_pre_optimizer_step(None, trainer.state, None)

    rank_dir = capsule.writer.directory
    assert not (rank_dir / "capsule_000001.pt").exists()
    assert not (rank_dir / "capsule_000001.json").exists()


def test_payload_write_error_warns_and_allows_the_watchdog_to_run(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(1, 1, bias=False)

    class Watchdog:
        def __init__(self, model):
            self.model = model

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            del args, state, kwargs
            for parameter in self.model.parameters():
                parameter.grad = None
            return control

    class Trainer:
        def __init__(self):
            self.model = Model()
            self.state = SimpleNamespace(global_step=0)
            self.callback_handler = SimpleNamespace(callbacks=[Watchdog(self.model)])

        def add_callback(self, callback):
            self.callback_handler.callbacks.append(callback)

        def _compute_loss(self, model, inputs):
            return (model.lora_A(inputs["input_ids"]) * inputs["poison"]).sum()

    trainer = Trainer()
    capsule = harness.Issue342CrashCapsule(
        trainer,
        output_dir=tmp_path / "writer-error",
        max_total_bytes=1 << 20,
        acknowledge_training_rows=True,
        instrument_lora_operands=False,
    ).install()
    trainer._compute_loss(
        trainer.model,
        {
            "input_ids": torch.ones((1, 1)),
            "poison": torch.tensor([[float("nan")]]),
        },
    ).backward()
    monkeypatch.setattr(
        capsule.writer,
        "write_torch",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("simulated disk-full")),
    )

    with pytest.warns(RuntimeWarning, match="continuing to Soup's watchdog"):
        for callback in trainer.callback_handler.callbacks:
            callback.on_pre_optimizer_step(None, trainer.state, None)

    assert trainer.model.lora_A.weight.grad is None
    assert not (capsule.writer.directory / "capsule_000001.pt").exists()


def test_install_fails_closed_when_callback_order_cannot_be_verified(tmp_path):
    torch = pytest.importorskip("torch")
    harness = _module()

    class Trainer:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            self.state = SimpleNamespace(global_step=0)

        def _compute_loss(self, model, inputs):
            return model(inputs["input_ids"]).sum()

        def add_callback(self, callback):
            del callback

    trainer = Trainer()
    original_loss = trainer._compute_loss
    capsule = harness.Issue342CrashCapsule(
        trainer,
        output_dir=tmp_path / "unsupported-trainer",
        max_total_bytes=1 << 20,
        acknowledge_training_rows=True,
        instrument_lora_operands=False,
    )

    with pytest.raises(harness.CapsuleError, match="callback_handler.callbacks"):
        capsule.install()

    assert trainer._compute_loss == original_loss
    assert capsule._closed is True


def test_unknown_sdpa_mode_is_rejected_explicitly():
    harness = _module()

    with pytest.raises(ValueError, match="unknown SDPA mode"):
        with harness.sdpa_backend("not-a-backend"):
            pass


def test_sdpa_baseline_leaves_the_native_policy_untouched(monkeypatch):
    harness = _module()

    def must_not_import_torch():
        raise AssertionError("baseline must not force an SDPA backend")

    monkeypatch.setattr(harness, "_torch", must_not_import_torch)

    with harness.sdpa_backend("baseline"):
        pass


def test_cuda_sync_records_the_observed_boundary_before_reraising(tmp_path):
    harness = _module()
    writer = harness.CapsuleWriter.create(tmp_path / "sync", max_total_bytes=4096)

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            raise RuntimeError("simulated asynchronous device error")

    capsule = object.__new__(harness.Issue342CrashCapsule)
    capsule.torch = SimpleNamespace(cuda=Cuda())
    capsule.writer = writer
    capsule._serial = 4
    capsule.rank = 0
    capsule.world_size = 1

    with pytest.raises(RuntimeError, match="simulated asynchronous"):
        capsule._synchronize("before_loss")

    error = json.loads((writer.directory / "cuda_error_000004_before_loss.json").read_text())
    assert error["boundary"] == "before_loss"


def test_cuda_sync_preserves_its_error_when_the_error_record_write_fails(monkeypatch, tmp_path):
    harness = _module()
    writer = harness.CapsuleWriter.create(tmp_path / "sync-write-error", max_total_bytes=4096)

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            raise RuntimeError("simulated asynchronous device error")

    capsule = object.__new__(harness.Issue342CrashCapsule)
    capsule.torch = SimpleNamespace(cuda=Cuda())
    capsule.writer = writer
    capsule._serial = 4
    capsule.rank = 0
    capsule.world_size = 1
    monkeypatch.setattr(
        writer,
        "write_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("simulated disk-full")),
    )

    with pytest.warns(RuntimeWarning, match="preserving the original device error"):
        with pytest.raises(RuntimeError, match="simulated asynchronous"):
            capsule._synchronize("before_loss")


def test_sdpa_prefers_the_current_torch_attention_context_manager(monkeypatch):
    harness = _module()
    calls = []

    @contextmanager
    def sdpa_kernel(backends):
        calls.append(backends)
        yield

    backend_type = SimpleNamespace(MATH="math", FLASH_ATTENTION="flash")
    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(
            attention=SimpleNamespace(sdpa_kernel=sdpa_kernel, SDPBackend=backend_type)
        )
    )
    monkeypatch.setattr(harness, "_torch", lambda: fake_torch)

    with harness.sdpa_backend("math"):
        pass

    assert calls == [["math"]]


def test_sdpa_current_api_refuses_an_unavailable_requested_backend(monkeypatch):
    harness = _module()
    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(
            attention=SimpleNamespace(sdpa_kernel=lambda _: None, SDPBackend=object())
        )
    )
    monkeypatch.setattr(harness, "_torch", lambda: fake_torch)

    with pytest.raises(RuntimeError, match="does not expose"):
        with harness.sdpa_backend("flash"):
            pass


def test_sdpa_falls_back_to_the_legacy_context_manager_when_needed(monkeypatch):
    harness = _module()
    calls = []

    @contextmanager
    def legacy(**kwargs):
        calls.append(kwargs)
        yield

    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(),
        backends=SimpleNamespace(cuda=SimpleNamespace(sdp_kernel=legacy)),
    )
    monkeypatch.setattr(harness, "_torch", lambda: fake_torch)

    with harness.sdpa_backend("mem_efficient"):
        pass

    assert calls == [
        {
            "enable_flash": False,
            "enable_math": False,
            "enable_mem_efficient": True,
            "enable_cudnn": False,
        }
    ]


def test_sdpa_legacy_context_omits_an_unsupported_cudnn_keyword(monkeypatch):
    harness = _module()
    calls = []

    @contextmanager
    def legacy(enable_flash, enable_math, enable_mem_efficient):
        calls.append((enable_flash, enable_math, enable_mem_efficient))
        yield

    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(),
        backends=SimpleNamespace(cuda=SimpleNamespace(sdp_kernel=legacy)),
    )
    monkeypatch.setattr(harness, "_torch", lambda: fake_torch)

    with harness.sdpa_backend("math"):
        pass

    assert calls == [(False, True, False)]
