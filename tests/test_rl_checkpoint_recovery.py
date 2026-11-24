"""A controller rollback requires a complete policy recovery (#3305)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_rl_checkpoint_distributed import _fixture, _same


@pytest.fixture(autouse=True)
def training_dependencies():
    for module in ("torch", "peft", "transformers"):
        pytest.importorskip(module, reason=f"RL recovery tests require {module}")


def _case(tmp_path, monkeypatch, *, task="grpo", **training):
    import torch

    if task == "ppo":
        pytest.importorskip("trl", reason="PPO policy wrapper requires trl")
    monkeypatch.chdir(tmp_path)
    model, optimizer, trainer, callbacks = _fixture(0, task=task, **training)
    checkpoint, controller = callbacks[:2]
    checkpoint.save_checkpoint(step=1, model=model, optimizer=optimizer)
    saved = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
    assert saved[1]["state"]
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(3)
        for slots in optimizer.state.values():
            slots["exp_avg"].add_(7)
            slots["exp_avg_sq"].add_(9)
            slots["step"].add_(2)
    optimizer.param_groups[0]["lr"] = 0.9
    controller._last_good_step = 1
    controller._observe = lambda snapshot, step: ({}, {"info_rm": 1.0})
    return model, optimizer, trainer, checkpoint, controller, saved


def _run(model, optimizer, trainer, controller):
    from transformers import TrainerControl

    control = TrainerControl()
    returned = controller.on_step_end(
        trainer.args, SimpleNamespace(global_step=2), control,
        model=trainer.model, optimizer=optimizer,
    )
    assert returned is control
    record = json.loads(controller.log_writer.path.read_text().splitlines()[-1])
    return control, record


@pytest.mark.parametrize("task", ["grpo", "ppo"])
def test_intact_controller_recovery_matches_saved_state_and_next_update(
    tmp_path, monkeypatch, task,
):
    import torch

    model, optimizer, trainer, _checkpoint, controller, saved = _case(
        tmp_path, monkeypatch, task=task,
    )
    reference = copy.deepcopy(model)
    reference.load_state_dict(saved[0])
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    reference_optimizer.load_state_dict(saved[1])
    control, record = _run(model, optimizer, trainer, controller)
    assert not control.should_training_stop
    assert record["escalation"] == "rollback to step 1 (restored=True)"
    assert _same(model.state_dict(), saved[0])
    assert _same(optimizer.state_dict(), saved[1])
    for policy, opt in ((model, optimizer), (reference, reference_optimizer)):
        opt.zero_grad()
        for parameter in policy.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 0.25)
        opt.step()
    assert _same(model.state_dict(), reference.state_dict())
    assert _same(optimizer.state_dict(), reference_optimizer.state_dict())


def _damage(kind):
    import torch
    from safetensors.torch import load_file, save_file

    directory = Path("run/rl-checkpoints/step-1")
    adapter = directory / "adapter_model.safetensors"
    optimizer = directory / "optimizer.pt"
    manifest = directory / "manifest.json"
    if kind in {"adapter_corrupt", "optimizer_corrupt", "manifest_corrupt"}:
        {"adapter_corrupt": adapter, "optimizer_corrupt": optimizer,
         "manifest_corrupt": manifest}[kind].write_bytes(b"broken")
    elif kind in {"adapter_missing", "optimizer_missing", "manifest_missing"}:
        {"adapter_missing": adapter, "optimizer_missing": optimizer,
         "manifest_missing": manifest}[kind].unlink()
    elif kind in {"adapter_keys", "adapter_shape"}:
        weights = load_file(str(adapter))
        name = next(iter(weights))
        if kind == "adapter_keys":
            del weights[name]
        else:
            weights[name] = weights[name][:1]
        save_file(weights, str(adapter))
    elif kind.startswith("optimizer_"):
        state = torch.load(optimizer, weights_only=True)
        if kind == "optimizer_groups":
            state["param_groups"].append(copy.deepcopy(state["param_groups"][0]))
        elif kind == "optimizer_parameters":
            state["param_groups"][0]["params"].pop()
        elif kind == "optimizer_slots":
            next(iter(state["state"].values()))["exp_avg"] = torch.zeros(1)
        elif kind == "optimizer_fields":
            del next(iter(state["state"].values()))["exp_avg"]
        elif kind == "optimizer_structure":
            state["state"] = []
        elif kind == "optimizer_nonplain":
            state["extra"] = torch.float32
        else:
            raise AssertionError(kind)
        torch.save(state, optimizer)
    else:
        record = json.loads(manifest.read_text())
        changes = {"manifest_step": {"step": 2}, "manifest_task": {"task": "ppo"},
                   "manifest_path": {"checkpoint_dir": "elsewhere"}}
        record.update(changes[kind])
        manifest.write_text(json.dumps(record))


@pytest.mark.parametrize("damage", [
    "adapter_corrupt", "adapter_missing", "adapter_keys", "adapter_shape",
    "optimizer_corrupt", "optimizer_missing", "optimizer_groups", "optimizer_parameters",
    "optimizer_slots", "optimizer_fields", "optimizer_structure", "optimizer_nonplain",
    "manifest_missing", "manifest_corrupt", "manifest_step", "manifest_task", "manifest_path",
])
def test_invalid_required_state_stops_before_mutating_live_objects(
    tmp_path, monkeypatch, caplog, damage,
):
    model, optimizer, trainer, _checkpoint, controller, _saved = _case(tmp_path, monkeypatch)
    _damage(damage)
    before = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
    control, record = _run(model, optimizer, trainer, controller)
    assert control.should_training_stop
    assert record["escalation"] == "rollback to step 1 (restored=False)"
    assert record["recovery"]["adapter"] is False
    assert record["recovery"]["optimizer"] is False
    assert record["recovery"]["error"]
    assert "RL rollback to step 1 failed" in caplog.text
    assert controller._state.recovery_attempts == 1
    assert _same(model.state_dict(), before[0])
    assert _same(optimizer.state_dict(), before[1])


@pytest.mark.parametrize("mode", ["config_off", "manifest_adapter_only"])
def test_intentional_adapter_only_recovery_does_not_read_optimizer(
    tmp_path, monkeypatch, mode,
):
    from soup_cli.utils import rl_checkpoint

    model, optimizer, trainer, checkpoint, controller, saved = _case(tmp_path, monkeypatch)
    if mode == "config_off":
        checkpoint.config = replace(checkpoint.config, include_optimizer_state=False)
    else:
        path = Path("run/rl-checkpoints/step-1/manifest.json")
        record = json.loads(path.read_text())
        record["has_optimizer"] = False
        path.write_text(json.dumps(record))
        Path("run/rl-checkpoints/step-1/optimizer.pt").unlink()
    before = copy.deepcopy(optimizer.state_dict())
    def refuse_read(path):
        raise AssertionError("adapter-only recovery must not read optimizer state")
    monkeypatch.setattr(rl_checkpoint, "load_optimizer_state", refuse_read)
    control, record = _run(model, optimizer, trainer, controller)
    assert not control.should_training_stop
    assert record["escalation"] == "rollback to step 1 (restored=True)"
    assert record["recovery"] == {"adapter": True, "optimizer": False, "required": False}
    assert _same(model.state_dict(), saved[0])
    assert _same(optimizer.state_dict(), before)


@pytest.mark.parametrize("component", ["adapter", "optimizer"])
def test_application_failure_reports_component_outcomes_and_stops(
    tmp_path, monkeypatch, component,
):
    import peft

    model, optimizer, trainer, _checkpoint, controller, saved = _case(tmp_path, monkeypatch)
    owner, name = (peft, "set_peft_model_state_dict") if component == "adapter" else (
        optimizer, "load_state_dict"
    )
    original = getattr(owner, name)
    def apply_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected application failure\x1b[31m")
    monkeypatch.setattr(owner, name, apply_then_fail)
    control, record = _run(model, optimizer, trainer, controller)
    assert control.should_training_stop
    assert record["escalation"] == "rollback to step 1 (restored=False)"
    assert record["recovery"]["adapter"] is (component == "optimizer")
    assert record["recovery"]["optimizer"] is False
    assert "injected application failure" in record["recovery"]["error"]
    assert "\x1b" not in record["recovery"]["error"]
    assert _same(model.state_dict(), saved[0])


def test_recovery_keeps_elevated_pid_state_and_reports_complete_components(tmp_path, monkeypatch):
    from transformers import TrainerControl

    model, optimizer, _trainer, _checkpoint, controller, _saved = _case(tmp_path, monkeypatch)
    controller._state = replace(controller._state, beta=0.7, integral=0.25)
    telemetry = {}
    control = TrainerControl()
    controller._escalate(model, optimizer, control, telemetry)
    assert controller._state.beta == 0.7
    assert controller._state.integral == 0.25
    assert controller._state.recovery_attempts == 1
    assert not control.should_training_stop
    assert telemetry["recovery"] == {"adapter": True, "optimizer": True, "required": True}


def test_missing_live_optimizer_cannot_apply_adapter_first(tmp_path, monkeypatch):
    model, optimizer, trainer, _checkpoint, controller, _saved = _case(tmp_path, monkeypatch)
    before = copy.deepcopy(model.state_dict())
    control, record = _run(model, None, trainer, controller)
    assert control.should_training_stop
    assert record["escalation"] == "rollback to step 1 (restored=False)"
    assert "requires a live optimizer" in record["recovery"]["error"]
    assert _same(model.state_dict(), before)


def test_preloaded_optimizer_is_not_read_again_after_adapter_application(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from soup_cli.utils import rl_checkpoint

    model, optimizer, trainer, _checkpoint, controller, saved = _case(tmp_path, monkeypatch)
    loader = Mock(side_effect=[copy.deepcopy(saved[1]), OSError("second read must not happen")])
    monkeypatch.setattr(rl_checkpoint, "load_optimizer_state", loader)
    control, record = _run(model, optimizer, trainer, controller)
    assert not control.should_training_stop
    assert record["escalation"] == "rollback to step 1 (restored=True)"
    assert loader.call_count == 1
    assert _same(optimizer.state_dict(), saved[1])


@pytest.mark.parametrize("kind", ["sgd", "adafactor", "fresh_adamw"])
def test_strict_recovery_preserves_supported_optimizer_layouts(tmp_path, monkeypatch, kind):
    import torch

    model, _optimizer, trainer, checkpoint, controller, saved = _case(tmp_path, monkeypatch)
    if kind == "fresh_adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.9)
    else:
        if kind == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        else:
            from transformers.optimization import Adafactor

            optimizer = Adafactor(
                model.parameters(), lr=0.01, relative_step=False, scale_parameter=False,
            )
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        checkpoint.save_checkpoint(step=1, model=model, optimizer=optimizer)
        saved = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
        optimizer.param_groups[0]["lr"] = 0.9
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(3)
    control, record = _run(model, optimizer, trainer, controller)
    assert not control.should_training_stop
    assert record["escalation"] == "rollback to step 1 (restored=True)"
    assert _same(model.state_dict(), saved[0])
    assert _same(optimizer.state_dict(), saved[1])


@pytest.mark.parametrize("outcome", [
    {"adapter": False, "optimizer": True, "required": True},
    {"adapter": True, "optimizer": False, "required": True},
    {"adapter": False, "optimizer": False, "required": False},
    {"adapter": True, "optimizer": True, "required": True, "error": "application failed"},
])
def test_controller_never_counts_incomplete_or_failed_outcome_as_success(
    tmp_path, monkeypatch, outcome,
):
    model, optimizer, trainer, checkpoint, controller, _saved = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(checkpoint, "_restore_components", lambda **kwargs: dict(outcome))
    control, record = _run(model, optimizer, trainer, controller)
    assert control.should_training_stop
    assert record["escalation"] == "rollback to step 1 (restored=False)"
    assert record["recovery"]["adapter"] == outcome["adapter"]
    assert record["recovery"]["optimizer"] == outcome["optimizer"]
    assert record["recovery"]["error"]


def test_real_trainer_stops_before_a_second_update_after_required_recovery_failure(
    tmp_path, monkeypatch,
):
    from transformers import Trainer, TrainingArguments

    model, optimizer, _trainer, _checkpoint, controller, _saved = _case(tmp_path, monkeypatch)
    _damage("adapter_corrupt")
    args = TrainingArguments(
        output_dir="trainer", use_cpu=True, max_steps=3, per_device_train_batch_size=1,
        save_strategy="no", logging_strategy="no", report_to=[], disable_tqdm=True,
        dataloader_pin_memory=False,
    )
    row = {"input_ids": [1, 2, 3], "labels": [1, 2, 3]}
    trainer = Trainer(
        model=model, args=args, train_dataset=[row] * 3, callbacks=[controller],
        optimizers=(optimizer, None),
    )
    trainer.beta = trainer.args.beta = 0.02
    controller.attach(trainer)
    trainer.train()
    assert trainer.state.global_step == 1
    record = json.loads(controller.log_writer.path.read_text().splitlines()[-1])
    assert record["step"] == 1
    assert record["escalation"] == "rollback to step 1 (restored=False)"


@pytest.mark.requires_symlink
def test_symlinked_optimizer_is_refused_before_adapter_mutation(tmp_path, monkeypatch):
    model, optimizer, trainer, _checkpoint, controller, _saved = _case(tmp_path, monkeypatch)
    path = Path("run/rl-checkpoints/step-1/optimizer.pt")
    destination = tmp_path / "optimizer.pt"
    path.rename(destination)
    path.symlink_to(destination)
    before = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
    control, record = _run(model, optimizer, trainer, controller)
    assert control.should_training_stop
    assert "symlink" in record["recovery"]["error"]
    assert _same(model.state_dict(), before[0])
    assert _same(optimizer.state_dict(), before[1])
